// io.hpp — read the .sbcsr sparse-binary corpus and write the .somw codebook.
//
// These two binary formats are shared with SparseBinSOM so the same evaluation harness
// (SparseBinEval) and the common cosine metrics tool can score StandardSparseSOM's output
// against SparseBinSOM's on equal footing. StandardSparseSOM is a *baseline*: it trains the
// same kind of map but with the standard NVIDIA cuSPARSE SpMM in place of SparseBinSOM's
// bespoke binary/feature-major BMU kernel.
//
//   .sbcsr : magic "SBCSR1\0\0" (8B), uint32 n_samples,n_features,n_nonzeros,reserved;
//            uint32 row_ptr[n_samples+1]; uint16 col_idx[n_nonzeros]  (present feature ids).
//   .somw  : magic "SBSOMW1\0" (8B), uint32 map_rows,map_cols,n_features,reserved;
//            float W[n_features * map_rows * map_cols], feature-major  W[v*K + k].

#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ssom {

struct Corpus {
    uint32_t n_samples = 0, n_features = 0, n_nonzeros = 0;
    std::vector<uint32_t> row_ptr;   // [n_samples+1] CSR offsets
    std::vector<uint16_t> col_idx;   // [n_nonzeros]  present feature ids
};

inline Corpus load_sbcsr(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open corpus " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "SCSR1\0\0\0", 8) != 0 &&           // .scsr
        std::memcmp(magic, "SBCSR1\0\0", 8) != 0)              // .sbcsr (legacy)
        throw std::runtime_error("not a .scsr/.sbcsr: " + path);
    uint32_t h[4];
    f.read(reinterpret_cast<char*>(h), 16);
    Corpus c;
    c.n_samples = h[0]; c.n_features = h[1]; c.n_nonzeros = h[2];
    c.row_ptr.resize((size_t)c.n_samples + 1);
    c.col_idx.resize(c.n_nonzeros);
    f.read(reinterpret_cast<char*>(c.row_ptr.data()), (std::streamsize)(c.n_samples + 1) * 4);
    f.read(reinterpret_cast<char*>(c.col_idx.data()), (std::streamsize)c.n_nonzeros * 2);
    if (!f) throw std::runtime_error("truncated corpus " + path);
    return c;
}

// Save a feature-major codebook W[v*K + k] (K = map_rows*map_cols) as .somw.
inline void save_somw(const std::string& path, uint32_t map_rows, uint32_t map_cols,
                      uint32_t n_features, const std::vector<float>& W) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path + " for writing");
    char magic[8] = {'S', 'B', 'S', 'O', 'M', 'W', '1', '\0'};
    uint32_t h[4] = {map_rows, map_cols, n_features, 0};
    f.write(magic, 8);
    f.write(reinterpret_cast<const char*>(h), 16);
    f.write(reinterpret_cast<const char*>(W.data()), (std::streamsize)W.size() * sizeof(float));
    if (!f) throw std::runtime_error("write failed for " + path);
}

} // namespace ssom
