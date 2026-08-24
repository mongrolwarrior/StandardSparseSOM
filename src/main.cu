// StandardSparseSOM — a baseline sparse SOM that uses NVIDIA cuSPARSE for the BMU step,
// in place of SparseBinSOM's bespoke binary/feature-major kernel. Same GPU, same dense
// codebook, same .sbcsr input / .somw output, so the two can be compared head-to-head
// (wall time + cosine QE/TE via the shared metrics tool).
//
// It ALSO carries both codebook layouts as a --layout option, for an internal
// feature-major vs node-major comparison (this is a benchmark, not a generic library, so
// keeping both in one binary is fine):
//   feature-major  W[v*K + k]  — V×K, neuron-contiguous per feature (cuSPARSE ORDER_ROW)
//   node-major     W[k*V + v]  — K×V, feature-contiguous per neuron (cuSPARSE ORDER_COL)
// Both feed cusparseSpMM as the same logical V×K dense operand, differing only in order/ld.
//
// Per epoch:
//   BMU      scores = X · W via cusparseSpMM (X = samples×V sparse-binary CSR, tiled over
//            samples because the dense scores block can't be materialised whole), then an
//            argmax kernel → argmax(2·x·w − ‖w‖²) = nearest neuron.
//   Update   batch SOM: scatter each sample's features into a per-BMU sum S, hit count;
//            blur both over the 2D map grid with a separable Gaussian; W = blur(S)/blur(hit).
//
// cuSPARSE only does the BMU SpMM (it doesn't exploit binary values and materialises a dense
// scores tile — the two costs the comparison exposes); the rest is plain CUDA.
//
// --precision fp16|fp32 controls codebook storage. FP16 halves the codebook's DRAM footprint
// and bandwidth, matching SparseBinarySOM's __half codebook for a fair comparison.
// Accumulators and blur always run in FP32.

#include "io.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cusparse.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
  std::exit(1);} } while(0)
#define CUSPARSE_CHECK(x) do { cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuSPARSE error %d at %s:%d\n",(int)s,__FILE__,__LINE__); \
  std::exit(1);} } while(0)

namespace ssom {

template<typename T> __device__ __forceinline__ float to_float(T x);
template<> __device__ __forceinline__ float to_float<float>(float x) { return x; }
template<> __device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }

template<typename T> __device__ __forceinline__ T from_float(float x);
template<> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template<> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

// Codebook (and per-BMU sum S) index for weight of feature v at neuron k. Templated on layout
// so each variant compiles to its own clean, coalescing-appropriate addressing.
template<bool NODE> __device__ __forceinline__
size_t widx(uint32_t v, uint32_t k, uint32_t V, uint32_t K) {
    return NODE ? (size_t)k * V + v : (size_t)v * K + k;
}

__global__ void rebase_rowptr(const int* __restrict__ full, int* __restrict__ out,
                              uint32_t r0, uint32_t tile_rows) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= tile_rows) out[i] = full[r0 + i] - full[r0];
}

template<bool NODE, typename W_T> __global__ void norms_kernel(const W_T* __restrict__ W,
                                                 float* __restrict__ norms, uint32_t K, uint32_t V) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    double acc = 0.0;
    for (uint32_t v = 0; v < V; ++v) { float w = to_float(W[widx<NODE>(v, k, V, K)]); acc += (double)w * w; }
    norms[k] = (float)acc;
}

// One dense scores tile (tile_rows×K, row-major) → bmu1/bmu2/dist. One thread per row.
template<typename C_T> __global__ void argmax_kernel(const C_T* __restrict__ C, const float* __restrict__ norms,
                              const uint32_t* __restrict__ row_ptr, uint32_t r0,
                              uint32_t tile_rows, uint32_t K,
                              uint32_t* __restrict__ bmu1, uint32_t* __restrict__ bmu2,
                              float* __restrict__ dist) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= tile_rows) return;
    const C_T* row = C + (size_t)r * K;
    float s1 = -1e30f, s2 = -1e30f, dot1 = 0.0f; int b1 = 0, b2 = 0;
    for (uint32_t k = 0; k < K; ++k) {
        float c_val = to_float(row[k]);
        float score = 2.0f * c_val - norms[k];
        if (score > s1) { s2 = s1; b2 = b1; s1 = score; b1 = (int)k; dot1 = c_val; }
        else if (score > s2) { s2 = score; b2 = (int)k; }
    }
    uint32_t g = r0 + r;
    float nnz = (float)(row_ptr[g + 1] - row_ptr[g]);
    bmu1[g] = (uint32_t)b1; bmu2[g] = (uint32_t)b2;
    dist[g] = sqrtf(fmaxf(0.0f, nnz - 2.0f * dot1 + norms[b1]));
}

template<bool NODE> __global__ void accumulate_kernel(const uint32_t* __restrict__ row_ptr,
                                  const int* __restrict__ col_idx, const uint32_t* __restrict__ bmu1,
                                  float* __restrict__ S, float* __restrict__ hit,
                                  uint32_t n_samples, uint32_t K, uint32_t V) {
    uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_samples) return;
    uint32_t k = bmu1[g];
    uint32_t a = row_ptr[g], b = row_ptr[g + 1];
    for (uint32_t j = a; j < b; ++j) atomicAdd(&S[widx<NODE>((uint32_t)col_idx[j], k, V, K)], 1.0f);
    atomicAdd(&hit[k], 1.0f);
}

// Separable Gaussian blur of `n_maps` grid-maps (each rows×cols) along one axis. For S,
// n_maps=V and Vp=V (a map per feature); for the hit map, n_maps=1 and Vp=1 (so widx → cell).
template<bool NODE> __global__ void blur_kernel(const float* __restrict__ in, float* __restrict__ out,
                            const float* __restrict__ gw, int radius,
                            uint32_t n_maps, uint32_t Vp, uint32_t rows, uint32_t cols, int axis) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t K = rows * cols;
    if (idx >= (size_t)n_maps * K) return;
    uint32_t m = idx / K, cell = idx % K;
    int gr = cell / cols, gc = cell % cols;
    double acc = 0.0;
    for (int t = -radius; t <= radius; ++t) {
        int rr = gr, cc = gc;
        if (axis == 0) { cc = gc + t; cc = cc < 0 ? 0 : (cc >= (int)cols ? cols - 1 : cc); }
        else           { rr = gr + t; rr = rr < 0 ? 0 : (rr >= (int)rows ? rows - 1 : rr); }
        acc += (double)gw[t + radius] * in[widx<NODE>(m, (uint32_t)(rr * cols + cc), Vp, K)];
    }
    out[widx<NODE>(m, cell, Vp, K)] = (float)acc;
}

// Topographic error: count samples whose 1st/2nd BMU are not 8-adjacent on the grid.
__global__ void te_kernel(const uint32_t* __restrict__ bmu1, const uint32_t* __restrict__ bmu2,
                          uint32_t n, uint32_t cols, unsigned long long* __restrict__ count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int dr = (int)(bmu1[i] / cols) - (int)(bmu2[i] / cols); dr = dr < 0 ? -dr : dr;
    int dc = (int)(bmu1[i] % cols) - (int)(bmu2[i] % cols); dc = dc < 0 ? -dc : dc;
    if ((dr > dc ? dr : dc) > 1) atomicAdd(count, 1ULL);
}

// Input-space distance between neurons p,q: sqrt(‖w_p‖²+‖w_q‖²−2 w_p·w_q).
template<bool NODE, typename W_T> __device__ float edge_dist(const W_T* W, const float* norms,
                                               uint32_t K, uint32_t V, uint32_t p, uint32_t q) {
    double dot = 0.0;
    for (uint32_t v = 0; v < V; ++v)
        dot += (double)to_float(W[widx<NODE>(v, p, V, K)]) * to_float(W[widx<NODE>(v, q, V, K)]);
    double d2 = (double)norms[p] + norms[q] - 2.0 * dot;
    return sqrtf((float)fmax(0.0, d2));
}

// Kaski-Lagus per monitored sample: bmu_dist + length of the capped L-path bmu1->bmu2.
template<bool NODE, typename W_T> __global__ void kl_kernel(const W_T* __restrict__ W, const float* __restrict__ norms,
                          const uint32_t* __restrict__ bmu1, const uint32_t* __restrict__ bmu2,
                          const float* __restrict__ dist, uint32_t n_mon, uint32_t K, uint32_t V,
                          uint32_t cols, int path_radius, float* __restrict__ out) {
    uint32_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_mon) return;
    uint32_t c1 = bmu1[a], c2 = bmu2[a];
    int r2 = (int)(c2 / cols), q2 = (int)(c2 % cols);
    double path = 0.0; uint32_t prev = c1; int steps = 0, rr = (int)(c1 / cols), qq = (int)(c1 % cols);
    while (rr != r2 && steps < path_radius) { rr += (r2 > rr) ? 1 : -1; uint32_t cur = (uint32_t)rr * cols + qq;
        path += edge_dist<NODE, W_T>(W, norms, K, V, prev, cur); prev = cur; ++steps; }
    while (qq != q2 && steps < path_radius) { qq += (q2 > qq) ? 1 : -1; uint32_t cur = (uint32_t)rr * cols + qq;
        path += edge_dist<NODE, W_T>(W, norms, K, V, prev, cur); prev = cur; ++steps; }
    out[a] = dist[a] + (float)path;
}

template<bool NODE, typename W_T> __global__ void divide_kernel(const float* __restrict__ Snum,
                              const float* __restrict__ hitden, W_T* __restrict__ W,
                              uint32_t K, uint32_t V) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)V * K) return;
    uint32_t k = NODE ? (uint32_t)(idx / V) : (uint32_t)(idx % K);   // recover the neuron of this slot
    float d = hitden[k];
    W[idx] = from_float<W_T>((d > 1e-12f) ? Snum[idx] / d : 0.0f);
}

} // namespace ssom

using namespace ssom;
static int igrid(size_t n, uint32_t blk) { return (int)((n + blk - 1) / blk); }

int main(int argc, char** argv) {
    std::string corpus_path, save_weights, layout_s = "feature", nbhd = "gaussian",
                stop = "fixed", precision_s = "fp32";
    uint32_t rows = 0, cols = 0, epochs = 10;
    float sigma_init = -1.0f, sigma_min = 0.5f;
    int box_passes = 3;
    double tile_mb = 1024.0;
    unsigned seed = 42;
    // KL-stop knobs (mirror SparseBinSOM defaults), used only when --stop kl.
    double sigma_rate = 0.3, gref = 0.01, qe_rel_thresh = 0.001, wd_path_frac = 0.25, sigma_accel = 2.0;
    int kl_window = 3, plateau_window = 3, path_radius = 8;
    float te_converge_max = 0.5f;
    uint32_t kl_monitor = 200000;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(argv[++i]); };
        if      (a == "--rows")        rows = std::stoul(nxt());
        else if (a == "--cols")        cols = std::stoul(nxt());
        else if (a == "--map")         { rows = cols = std::stoul(nxt()); }
        else if (a == "--epochs")      epochs = std::stoul(nxt());
        else if (a == "--sigma-init")  sigma_init = std::stof(nxt());
        else if (a == "--sigma-min")   sigma_min = std::stof(nxt());
        else if (a == "--tile-mb")     tile_mb = std::stod(nxt());
        else if (a == "--seed")        seed = std::stoul(nxt());
        else if (a == "--layout")      layout_s = nxt();
        else if (a == "--neighbourhood") nbhd = nxt();
        else if (a == "--box-passes")  box_passes = std::stoi(nxt());
        else if (a == "--stop")        stop = nxt();            // fixed | kl
        else if (a == "--precision")   precision_s = nxt();     // fp16 | fp32
        else if (a == "--sigma-rate")  sigma_rate = std::stod(nxt());
        else if (a == "--gref")        gref = std::stod(nxt());
        else if (a == "--kl-window")   kl_window = std::stoi(nxt());
        else if (a == "--kl-monitor")  kl_monitor = std::stoul(nxt());
        else if (a == "--te-converge-max") te_converge_max = std::stof(nxt());
        else if (a == "--path-radius") path_radius = std::stoi(nxt());
        else if (a == "--save-weights") save_weights = nxt();
        else if (corpus_path.empty())  corpus_path = a;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (corpus_path.empty() || !rows || !cols) {
        std::fprintf(stderr, "usage: standardsparsesom CORPUS.sbcsr --rows R --cols C "
                             "[--layout feature|node --precision fp16|fp32 --epochs E "
                             "--sigma-init S0 --sigma-min Smin --save-weights P]\n");
        return 2;
    }
    if (layout_s != "feature" && layout_s != "node") { std::fprintf(stderr, "--layout must be feature|node\n"); return 2; }
    if (nbhd != "gaussian" && nbhd != "box") { std::fprintf(stderr, "--neighbourhood must be gaussian|box\n"); return 2; }
    if (stop != "fixed" && stop != "kl") { std::fprintf(stderr, "--stop must be fixed|kl\n"); return 2; }
    if (precision_s != "fp16" && precision_s != "fp32") { std::fprintf(stderr, "--precision must be fp16|fp32\n"); return 2; }
    const bool node = (layout_s == "node");
    const bool box = (nbhd == "box");   // box-blur (P passes ≈ Gaussian) = SparseBinSOM's update
    const bool kl_stop = (stop == "kl"); // KL-driven sigma schedule + plateau/TE-gated convergence
    const bool fp16 = (precision_s == "fp16");
    if (sigma_init < 0) sigma_init = std::max(rows, cols) / 2.0f;

    Corpus c = load_sbcsr(corpus_path);
    const uint32_t V = c.n_features, K = rows * cols, N = c.n_samples;
    std::printf("StandardSparseSOM (cuSPARSE, %s-major, %s neighbourhood%s, %s): %u samples x %u "
                "features -> %ux%u = %u neurons, %u epochs, sigma %.1f->%.1f\n", layout_s.c_str(),
                nbhd.c_str(), box ? (" x" + std::to_string(box_passes) + " passes").c_str() : "",
                precision_s.c_str(),
                N, V, rows, cols, K, epochs, (double)sigma_init, (double)sigma_min);

    // Upload corpus (cuSPARSE needs 32-bit indices, so col_idx uint16 is widened to int).
    std::vector<int> col_i32(c.col_idx.begin(), c.col_idx.end());
    std::vector<int> rp_i32(c.row_ptr.begin(), c.row_ptr.end());
    uint32_t* d_row_ptr; int *d_col_i32, *d_rp_full;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(N + 1) * 4));
    CUDA_CHECK(cudaMalloc(&d_col_i32, (size_t)c.n_nonzeros * 4));
    CUDA_CHECK(cudaMalloc(&d_rp_full, (size_t)(N + 1) * 4));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, c.row_ptr.data(), (size_t)(N + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_i32, col_i32.data(), (size_t)c.n_nonzeros * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rp_full, rp_i32.data(), (size_t)(N + 1) * 4, cudaMemcpyHostToDevice));

    const size_t WN = (size_t)V * K;
    std::vector<float> hW(WN);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f / std::sqrt((float)V));
    for (auto& w : hW) w = uni(rng);

    // Codebook and scratch buffers.
    //   d_W:  FP32 buffer, always allocated — serves as codebook (FP32 mode) or blur scratch (FP16).
    //   d_Wh: FP16 buffer, allocated only in FP16 mode — the actual codebook for cuSPARSE.
    //   d_S:  FP32 accumulator, always allocated.
    float *d_W, *d_norms, *d_S, *d_hit, *d_hit2;
    __half *d_Wh = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, WN * 4));  CUDA_CHECK(cudaMalloc(&d_S, WN * 4));
    CUDA_CHECK(cudaMalloc(&d_norms, K * 4)); CUDA_CHECK(cudaMalloc(&d_hit, K * 4)); CUDA_CHECK(cudaMalloc(&d_hit2, K * 4));

    if (fp16) {
        CUDA_CHECK(cudaMalloc(&d_Wh, WN * 2));
        std::vector<__half> hWh(WN);
        for (size_t i = 0; i < WN; ++i) hWh[i] = __float2half(hW[i]);
        CUDA_CHECK(cudaMemcpy(d_Wh, hWh.data(), WN * 2, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_W, hW.data(), WN * 4, cudaMemcpyHostToDevice));
    }

    uint32_t *d_bmu1, *d_bmu2; float* d_dist;
    CUDA_CHECK(cudaMalloc(&d_bmu1, (size_t)N * 4)); CUDA_CHECK(cudaMalloc(&d_bmu2, (size_t)N * 4));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)N * 4));

    uint32_t tile_rows = (uint32_t)std::max<size_t>(1, (size_t)(tile_mb * 1048576) / ((size_t)K * 4));
    tile_rows = std::min(tile_rows, N);
    uint32_t n_tiles = (N + tile_rows - 1) / tile_rows;
    size_t max_tile_nnz = 0;
    for (uint32_t t = 0; t < n_tiles; ++t) {
        uint32_t r0 = t * tile_rows, r1 = std::min(r0 + tile_rows, N);
        max_tile_nnz = std::max(max_tile_nnz, (size_t)(c.row_ptr[r1] - c.row_ptr[r0]));
    }

    // Scores tile and ones buffer — FP16 or FP32 matching the codebook.
    const size_t scores_elem = (size_t)tile_rows * K;
    float *d_C = nullptr; __half *d_Ch = nullptr;
    float *d_ones = nullptr; __half *d_ones_h = nullptr;
    cudaDataType_t w_dtype = fp16 ? CUDA_R_16F : CUDA_R_32F;

    if (fp16) {
        CUDA_CHECK(cudaMalloc(&d_Ch, scores_elem * 2));
        CUDA_CHECK(cudaMalloc(&d_ones_h, max_tile_nnz * 2));
        std::vector<__half> ones_h(max_tile_nnz, __float2half(1.0f));
        CUDA_CHECK(cudaMemcpy(d_ones_h, ones_h.data(), max_tile_nnz * 2, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_C, scores_elem * 4));
        CUDA_CHECK(cudaMalloc(&d_ones, max_tile_nnz * 4));
        std::vector<float> ones(max_tile_nnz, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_ones, ones.data(), max_tile_nnz * 4, cudaMemcpyHostToDevice));
    }
    std::printf("  tiling BMU: %u tiles of <=%u rows (scores tile %.0f MiB, %s)\n",
                n_tiles, tile_rows,
                (double)tile_rows * K * (fp16 ? 2 : 4) / 1048576.0,
                precision_s.c_str());

    int*   d_rp;  CUDA_CHECK(cudaMalloc(&d_rp, (size_t)(tile_rows + 1) * 4));

    cusparseHandle_t handle; CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseDnMatDescr_t matW;
    CUSPARSE_CHECK(cusparseCreateDnMat(&matW, V, K, node ? V : K,
                                       fp16 ? (void*)d_Wh : (void*)d_W, w_dtype,
                                       node ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));
    const float one = 1.0f, zero = 0.0f;
    void* spmm_buf = nullptr; size_t spmm_buf_sz = 0;
    const uint32_t B = 256;

    auto bmu_pass = [&]() {
        if (fp16) {
            if (node) norms_kernel<true,  __half><<<igrid(K, B), B>>>(d_Wh, d_norms, K, V);
            else      norms_kernel<false, __half><<<igrid(K, B), B>>>(d_Wh, d_norms, K, V);
        } else {
            if (node) norms_kernel<true,  float><<<igrid(K, B), B>>>(d_W, d_norms, K, V);
            else      norms_kernel<false, float><<<igrid(K, B), B>>>(d_W, d_norms, K, V);
        }
        for (uint32_t t = 0; t < n_tiles; ++t) {
            uint32_t r0 = t * tile_rows, r1 = std::min(r0 + tile_rows, N), tr = r1 - r0;
            int tnnz = (int)(c.row_ptr[r1] - c.row_ptr[r0]);
            rebase_rowptr<<<igrid((size_t)tr + 1, B), B>>>(d_rp_full, d_rp, r0, tr);
            cusparseSpMatDescr_t matX;
            CUSPARSE_CHECK(cusparseCreateCsr(&matX, tr, V, tnnz, d_rp,
                d_col_i32 + c.row_ptr[r0],
                fp16 ? (void*)d_ones_h : (void*)d_ones,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, w_dtype));
            cusparseDnMatDescr_t matC;
            CUSPARSE_CHECK(cusparseCreateDnMat(&matC, tr, K, K,
                fp16 ? (void*)d_Ch : (void*)d_C,
                w_dtype, CUSPARSE_ORDER_ROW));
            size_t need = 0;
            CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matX, matW, &zero, matC, CUDA_R_32F,
                CUSPARSE_SPMM_ALG_DEFAULT, &need));
            if (need > spmm_buf_sz) { cudaFree(spmm_buf); CUDA_CHECK(cudaMalloc(&spmm_buf, need)); spmm_buf_sz = need; }
            CUSPARSE_CHECK(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matX, matW, &zero, matC, CUDA_R_32F,
                CUSPARSE_SPMM_ALG_DEFAULT, spmm_buf));
            if (fp16)
                argmax_kernel<__half><<<igrid(tr, B), B>>>(d_Ch, d_norms, d_row_ptr, r0, tr, K, d_bmu1, d_bmu2, d_dist);
            else
                argmax_kernel<float><<<igrid(tr, B), B>>>(d_C, d_norms, d_row_ptr, r0, tr, K, d_bmu1, d_bmu2, d_dist);
            cusparseDestroySpMat(matX); cusparseDestroyDnMat(matC);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // KL-stop machinery (used only when --stop kl): a monitored-sample KL + a TE measure.
    uint32_t kl_mon = std::min(kl_monitor, N);
    float* d_kl_out = nullptr; unsigned long long* d_te_count = nullptr;
    if (kl_stop) { CUDA_CHECK(cudaMalloc(&d_kl_out, (size_t)kl_mon * 4)); CUDA_CHECK(cudaMalloc(&d_te_count, 8)); }
    auto compute_kl = [&]() -> double {
        if (fp16) {
            if (node) kl_kernel<true,  __half><<<igrid(kl_mon, B), B>>>(d_Wh, d_norms, d_bmu1, d_bmu2, d_dist, kl_mon, K, V, cols, path_radius, d_kl_out);
            else      kl_kernel<false, __half><<<igrid(kl_mon, B), B>>>(d_Wh, d_norms, d_bmu1, d_bmu2, d_dist, kl_mon, K, V, cols, path_radius, d_kl_out);
        } else {
            if (node) kl_kernel<true,  float><<<igrid(kl_mon, B), B>>>(d_W, d_norms, d_bmu1, d_bmu2, d_dist, kl_mon, K, V, cols, path_radius, d_kl_out);
            else      kl_kernel<false, float><<<igrid(kl_mon, B), B>>>(d_W, d_norms, d_bmu1, d_bmu2, d_dist, kl_mon, K, V, cols, path_radius, d_kl_out);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> h(kl_mon); CUDA_CHECK(cudaMemcpy(h.data(), d_kl_out, (size_t)kl_mon * 4, cudaMemcpyDeviceToHost));
        double s = 0.0; for (float x : h) s += x; return s / kl_mon;
    };
    auto compute_te = [&]() -> double {
        CUDA_CHECK(cudaMemset(d_te_count, 0, 8));
        te_kernel<<<igrid(N, B), B>>>(d_bmu1, d_bmu2, N, cols, d_te_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long cnt; CUDA_CHECK(cudaMemcpy(&cnt, d_te_count, 8, cudaMemcpyDeviceToHost));
        return (double)cnt / N;
    };

    double bmu_secs = 0.0, upd_secs = 0.0;
    double s_prog = 0.0, prev_stop = 0.0, path_peak = 0.0, rate = sigma_rate;
    uint32_t flat_streak = 0, ep_run = 0; bool converged = false;
    std::vector<double> kl_hist;
    float sigma_small = std::max(1.0f, 2.0f * sigma_min);
    auto wall0 = std::chrono::steady_clock::now();
    for (uint32_t ep = 0; ep < epochs; ++ep) {
        ep_run = ep + 1;
        auto a0 = std::chrono::steady_clock::now();
        bmu_pass();
        bmu_secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count();

        std::vector<float> hdist(N); CUDA_CHECK(cudaMemcpy(hdist.data(), d_dist, (size_t)N * 4, cudaMemcpyDeviceToHost));
        double qe = 0.0; for (float d : hdist) qe += d; qe /= N;

        float sigma; double kl = 0.0, te = 0.0;
        if (kl_stop) {
            kl = compute_kl(); te = compute_te();
            if (ep >= 1) {                              // advance progress by windowed KL improvement
                uint32_t w = std::min((uint32_t)kl_window, ep);
                double ref = kl_hist[ep - w];
                double gwin = ref > 1e-12 ? std::max(0.0, (ref - kl) / ref) : 0.0;
                s_prog += 1.0 - gwin / (gwin + gref);
            }
            kl_hist.push_back(kl);
            double span = std::max(0.0, (double)sigma_init - sigma_min);
            sigma = (float)(sigma_min + span * std::exp(-rate * s_prog));   // endpoint interpolation
            std::printf("epoch %3u  sigma=%.2f  QE=%.4f  KL=%.4f  TE=%.3f\n", ep, (double)sigma, qe, kl, te);
        } else {
            float frac = epochs > 1 ? (float)ep / (epochs - 1) : 1.0f;
            sigma = sigma_init * std::pow(sigma_min / sigma_init, frac);
            std::printf("epoch %3u  sigma=%.2f  QE=%.4f\n", ep, (double)sigma, qe);
        }
        std::fflush(stdout);

        if (kl_stop) {                                  // plateau / TE-gated convergence + accel watchdog
            double kl_path = kl - qe; path_peak = std::max(path_peak, kl_path);
            if (ep >= 1) {
                double rel = prev_stop > 1e-12 ? std::fabs(kl - prev_stop) / prev_stop
                                               : (std::fabs(kl - prev_stop) < 1e-12 ? 0.0 : 1.0);
                if (rel < qe_rel_thresh) ++flat_streak; else flat_streak = 0;
            }
            prev_stop = kl;
            if (flat_streak >= (uint32_t)plateau_window) {
                if (sigma <= sigma_small) {
                    bool path_ok = path_peak <= 1e-9 || kl_path <= wd_path_frac * path_peak;
                    if (path_ok && te <= te_converge_max) {
                        std::printf("converged after epoch %u (KL plateau; sigma=%.2f, TE=%.3f<=%.2f)\n",
                                    ep, (double)sigma, te, (double)te_converge_max);
                        converged = true;
                    } else {
                        std::printf("stopped after epoch %u: topology unresolved (TE=%.3f, path=%.4f)\n",
                                    ep, te, kl_path);
                    }
                    break;
                }
                rate *= sigma_accel; flat_streak = 0;
                std::printf("  watchdog: plateau at large sigma=%.2f -> rate %.3g\n", (double)sigma, rate);
            }
        }

        auto u0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemset(d_S, 0, WN * 4)); CUDA_CHECK(cudaMemset(d_hit, 0, K * 4));
        if (node) accumulate_kernel<true ><<<igrid(N, B), B>>>(d_row_ptr, d_col_i32, d_bmu1, d_S, d_hit, N, K, V);
        else      accumulate_kernel<false><<<igrid(N, B), B>>>(d_row_ptr, d_col_i32, d_bmu1, d_S, d_hit, N, K, V);
        // Neighbourhood weights. Gaussian: one pass, taps ~exp(-t^2/2sigma^2), radius ceil(3sigma). Box: a
        // uniform half-width-sigma kernel applied box_passes times per axis (approx Gaussian by the central
        // limit) — this is exactly SparseBinSOM's factorized update, for the fair-quality isolation.
        int radius, passes;
        std::vector<float> gw;
        if (box) {
            radius = std::max(1, (int)std::lround(sigma)); passes = box_passes;
            gw.assign(2 * radius + 1, 1.0f / (2 * radius + 1));
        } else {
            radius = std::max(1, (int)std::ceil(3.0f * sigma)); passes = 1;
            gw.resize(2 * radius + 1); double gsum = 0;
            for (int t = -radius; t <= radius; ++t) { gw[t + radius] = std::exp(-0.5 * (t * t) / (sigma * sigma)); gsum += gw[t + radius]; }
            for (auto& x : gw) x /= gsum;
        }
        float* d_gw; CUDA_CHECK(cudaMalloc(&d_gw, gw.size() * 4));
        CUDA_CHECK(cudaMemcpy(d_gw, gw.data(), gw.size() * 4, cudaMemcpyHostToDevice));
        auto blur1 = [&](float* in, float* out, uint32_t n_maps, uint32_t Vp, int axis) {
            int g = igrid((size_t)n_maps * K, B);
            if (node) blur_kernel<true ><<<g, B>>>(in, out, d_gw, radius, n_maps, Vp, rows, cols, axis);
            else      blur_kernel<false><<<g, B>>>(in, out, d_gw, radius, n_maps, Vp, rows, cols, axis);
        };
        // Separable: `passes` blurs along columns then rows, ping-ponging; 2·passes is even so
        // the result lands back in `buf`.
        auto sep_blur = [&](float* buf, float* tmp, uint32_t n_maps, uint32_t Vp) {
            float* a = buf; float* b = tmp;
            for (int axis = 0; axis < 2; ++axis)
                for (int pp = 0; pp < passes; ++pp) { blur1(a, b, n_maps, Vp, axis); std::swap(a, b); }
        };
        sep_blur(d_S, d_W, V, V);          // d_W is blur scratch (FP32 in both modes)
        sep_blur(d_hit, d_hit2, 1, 1);
        if (fp16) {
            if (node) divide_kernel<true,  __half><<<igrid(WN, B), B>>>(d_S, d_hit, d_Wh, K, V);
            else      divide_kernel<false, __half><<<igrid(WN, B), B>>>(d_S, d_hit, d_Wh, K, V);
        } else {
            if (node) divide_kernel<true,  float><<<igrid(WN, B), B>>>(d_S, d_hit, d_W, K, V);
            else      divide_kernel<false, float><<<igrid(WN, B), B>>>(d_S, d_hit, d_W, K, V);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_gw);
        upd_secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - u0).count();
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    std::printf("done: %u epochs in %.2fs  (BMU %.2fs, update %.2fs)  [%s-major, %s%s]\n",
                ep_run, secs, bmu_secs, upd_secs, layout_s.c_str(), precision_s.c_str(),
                kl_stop ? (converged ? ", converged" : ", stopped") : "");

    if (!save_weights.empty()) {
        if (fp16) {
            std::vector<__half> hWh(WN);
            CUDA_CHECK(cudaMemcpy(hWh.data(), d_Wh, WN * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < WN; ++i) hW[i] = __half2float(hWh[i]);
        } else {
            CUDA_CHECK(cudaMemcpy(hW.data(), d_W, WN * 4, cudaMemcpyDeviceToHost));
        }
        if (node) {                                 // .somw is feature-major; transpose K*V -> V*K
            std::vector<float> fm(WN);
            for (uint32_t k = 0; k < K; ++k)
                for (uint32_t v = 0; v < V; ++v) fm[(size_t)v * K + k] = hW[(size_t)k * V + v];
            save_somw(save_weights, rows, cols, V, fm);
        } else {
            save_somw(save_weights, rows, cols, V, hW);
        }
        std::printf("  wrote %s (%ux%u, %u features, feature-major .somw)\n",
                    save_weights.c_str(), rows, cols, V);
    }
    return 0;
}
