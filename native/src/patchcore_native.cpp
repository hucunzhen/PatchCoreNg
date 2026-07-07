#include "patchcore_native.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) { g_last_error = msg; }

float squared_distance_scalar(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#if defined(__AVX2__)
#include <immintrin.h>

float squared_distance_avx2(const float* a, const float* b, int dim) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    alignas(32) float buf[8];
    _mm256_store_ps(buf, acc);
    float sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

float squared_distance(const float* a, const float* b, int dim, bool use_simd) {
#if defined(__AVX2__)
    if (use_simd)
        return squared_distance_avx2(a, b, dim);
#else
    (void)use_simd;
#endif
    return squared_distance_scalar(a, b, dim);
}

void insert_topk_squared(float* topk, int k, float squared) {
    int max_idx = 0;
    for (int i = 1; i < k; ++i) {
        if (topk[i] > topk[max_idx])
            max_idx = i;
    }
    if (squared < topk[max_idx])
        topk[max_idx] = squared;
}

float average_topk(const float* topk, int k, int metric) {
    int valid = 0;
    for (int i = 0; i < k; ++i) {
        if (topk[i] < std::numeric_limits<float>::max())
            ++valid;
    }
    if (valid <= 0)
        return 0.0f;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        if (topk[i] >= std::numeric_limits<float>::max())
            continue;
        float v = topk[i];
        if (metric == PCN_SQUARED_EUCLIDEAN)
            v = std::sqrt(v);
        sum += v;
    }
    return sum / static_cast<float>(valid);
}

float score_query_bruteforce(
    const float* query,
    const float* bank,
    int bank_count,
    int dim,
    int k,
    int metric,
    bool use_simd) {
    std::vector<float> topk(k, std::numeric_limits<float>::max());
    for (int i = 0; i < bank_count; ++i) {
        const float* candidate = bank + static_cast<size_t>(i) * dim;
        const float sq = squared_distance(query, candidate, dim, use_simd);
        insert_topk_squared(topk.data(), k, sq);
    }
    return average_topk(topk.data(), k, metric);
}

struct AnnIndex {
    int cluster_count = 0;
    std::vector<float> centroids;
    std::vector<std::vector<int>> clusters;
};

void build_ann_index(
    const float* bank,
    int bank_count,
    int dim,
    int cluster_count,
    bool use_simd,
    AnnIndex* out) {
    cluster_count = std::max(2, std::min(cluster_count, bank_count));
    out->cluster_count = cluster_count;
    out->centroids.assign(static_cast<size_t>(cluster_count) * dim, 0.0f);
    out->clusters.assign(cluster_count, {});

    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(0, bank_count - 1);
    std::vector<int> used;
    used.reserve(cluster_count);

    for (int c = 0; c < cluster_count; ++c) {
        int pick = dist(rng);
        while (std::find(used.begin(), used.end(), pick) != used.end())
            pick = dist(rng);
        used.push_back(pick);
        std::memcpy(
            out->centroids.data() + static_cast<size_t>(c) * dim,
            bank + static_cast<size_t>(pick) * dim,
            static_cast<size_t>(dim) * sizeof(float));
    }

    for (int iter = 0; iter < 8; ++iter) {
        for (auto& cluster : out->clusters)
            cluster.clear();

        for (int i = 0; i < bank_count; ++i) {
            const float* emb = bank + static_cast<size_t>(i) * dim;
            int best = 0;
            float best_dist = std::numeric_limits<float>::max();
            for (int c = 0; c < cluster_count; ++c) {
                const float* centroid = out->centroids.data() + static_cast<size_t>(c) * dim;
                const float d = squared_distance(emb, centroid, dim, use_simd);
                if (d < best_dist) {
                    best_dist = d;
                    best = c;
                }
            }
            out->clusters[best].push_back(i);
        }

        for (int c = 0; c < cluster_count; ++c) {
            if (out->clusters[c].empty())
                continue;
            float* centroid = out->centroids.data() + static_cast<size_t>(c) * dim;
            std::fill(centroid, centroid + dim, 0.0f);
            for (int idx : out->clusters[c]) {
                const float* emb = bank + static_cast<size_t>(idx) * dim;
                for (int d = 0; d < dim; ++d)
                    centroid[d] += emb[d];
            }
            const float inv = 1.0f / static_cast<float>(out->clusters[c].size());
            for (int d = 0; d < dim; ++d)
                centroid[d] *= inv;
        }
    }
}

float score_query_ann(
    const float* query,
    const float* bank,
    int bank_count,
    int dim,
    int k,
    int metric,
    bool use_simd,
    const AnnIndex& ann,
    int probe_clusters) {
    probe_clusters = std::max(1, std::min(probe_clusters, ann.cluster_count));
    std::vector<std::pair<int, float>> ranked(ann.cluster_count);
    for (int c = 0; c < ann.cluster_count; ++c) {
        const float* centroid = ann.centroids.data() + static_cast<size_t>(c) * dim;
        ranked[c] = {c, squared_distance(query, centroid, dim, use_simd)};
    }
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + probe_clusters,
        ranked.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<float> topk(k, std::numeric_limits<float>::max());
    std::vector<char> visited(static_cast<size_t>(bank_count), 0);

    for (int p = 0; p < probe_clusters; ++p) {
        const int cluster_idx = ranked[p].first;
        for (int emb_idx : ann.clusters[cluster_idx]) {
            if (visited[emb_idx])
                continue;
            visited[emb_idx] = 1;
            const float* candidate = bank + static_cast<size_t>(emb_idx) * dim;
            const float sq = squared_distance(query, candidate, dim, use_simd);
            insert_topk_squared(topk.data(), k, sq);
        }
    }

    int count_valid = 0;
    for (int i = 0; i < k; ++i) {
        if (topk[i] < std::numeric_limits<float>::max())
            ++count_valid;
    }
    if (count_valid < k)
        return score_query_bruteforce(query, bank, bank_count, dim, k, metric, use_simd);

    return average_topk(topk.data(), k, metric);
}

void aggregate_patch(
    const float* feature_chw,
    int channels,
    int height,
    int width,
    int y,
    int x,
    int patch_size,
    float* out_patch) {
    const int half = patch_size / 2;
    std::fill(out_patch, out_patch + channels, 0.0f);
    int count = 0;

    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            const int ny = y + dy;
            const int nx = x + dx;
            if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                continue;
            for (int c = 0; c < channels; ++c) {
                const size_t idx = static_cast<size_t>(c) * height * width
                    + static_cast<size_t>(ny) * width + nx;
                out_patch[c] += feature_chw[idx];
            }
            ++count;
        }
    }

    if (count > 0) {
        const float inv = 1.0f / static_cast<float>(count);
        for (int c = 0; c < channels; ++c)
            out_patch[c] *= inv;
    }
}

int default_thread_count(int configured) {
    if (configured > 0)
        return configured;
    const unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0)
        return 4;
    return static_cast<int>(std::min(std::max(2u, hc), 16u));
}

} // namespace

struct PcnMemoryBank {
    std::vector<float> embeddings;
    int count = 0;
    int dim = 0;
    PcnBankConfig config{};
    AnnIndex ann;
};

extern "C" {

PCN_API PcnMemoryBank* pcn_bank_create(
    const float* embeddings,
    int bank_count,
    int dim,
    const PcnBankConfig* config) {
    if (!embeddings || bank_count <= 0 || dim <= 0 || !config) {
        set_error("Invalid bank create arguments.");
        return nullptr;
    }

    auto* bank = new (std::nothrow) PcnMemoryBank();
    if (!bank) {
        set_error("Out of memory.");
        return nullptr;
    }

    bank->count = bank_count;
    bank->dim = dim;
    bank->config = *config;
    bank->embeddings.assign(
        embeddings,
        embeddings + static_cast<size_t>(bank_count) * dim);

    if (config->use_ann && bank_count >= config->ann_cluster_count) {
        build_ann_index(
            bank->embeddings.data(),
            bank_count,
            dim,
            config->ann_cluster_count,
            config->use_simd != 0,
            &bank->ann);
    }

    return bank;
}

PCN_API void pcn_bank_destroy(PcnMemoryBank* bank) {
    delete bank;
}

PCN_API int pcn_aggregate_and_score(
    const PcnMemoryBank* bank,
    const float* feature_chw,
    int channels,
    int height,
    int width,
    const PcnScoreParams* params,
    float* out_distances,
    float* out_image_score) {
    if (!bank || !feature_chw || !params || !out_distances || !out_image_score) {
        set_error("Invalid score arguments.");
        return -1;
    }
    if (channels != bank->dim) {
        set_error("Feature channels do not match bank dimension.");
        return -2;
    }
    if (height <= 0 || width <= 0 || params->patch_size <= 0 || params->num_neighbors <= 0) {
        set_error("Invalid feature map or score parameters.");
        return -3;
    }

    const int patch_count = height * width;
    const int k = params->num_neighbors;
    const int threads = default_thread_count(params->patch_parallelism);

#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel
    {
        std::vector<float> patch(static_cast<size_t>(channels));
#pragma omp for schedule(static)
        for (int idx = 0; idx < patch_count; ++idx) {
            const int y = idx / width;
            const int x = idx % width;
            aggregate_patch(feature_chw, channels, height, width, y, x, params->patch_size, patch.data());

            float score = 0.0f;
            if (bank->config.use_ann && bank->ann.cluster_count > 0) {
                score = score_query_ann(
                    patch.data(),
                    bank->embeddings.data(),
                    bank->count,
                    bank->dim,
                    k,
                    bank->config.distance_metric,
                    bank->config.use_simd != 0,
                    bank->ann,
                    bank->config.ann_probe_clusters);
            } else {
                score = score_query_bruteforce(
                    patch.data(),
                    bank->embeddings.data(),
                    bank->count,
                    bank->dim,
                    k,
                    bank->config.distance_metric,
                    bank->config.use_simd != 0);
            }
            out_distances[idx] = score;
        }
    }
#else
    std::vector<float> patch(static_cast<size_t>(channels));
    for (int idx = 0; idx < patch_count; ++idx) {
        const int y = idx / width;
        const int x = idx % width;
        aggregate_patch(feature_chw, channels, height, width, y, x, params->patch_size, patch.data());
        float score = 0.0f;
        if (bank->config.use_ann && bank->ann.cluster_count > 0) {
            score = score_query_ann(
                patch.data(),
                bank->embeddings.data(),
                bank->count,
                bank->dim,
                k,
                bank->config.distance_metric,
                bank->config.use_simd != 0,
                bank->ann,
                bank->config.ann_probe_clusters);
        } else {
            score = score_query_bruteforce(
                patch.data(),
                bank->embeddings.data(),
                bank->count,
                bank->dim,
                k,
                bank->config.distance_metric,
                bank->config.use_simd != 0);
        }
        out_distances[idx] = score;
    }
#endif

    float max_score = 0.0f;
    for (int i = 0; i < patch_count; ++i) {
        if (out_distances[i] > max_score)
            max_score = out_distances[i];
    }
    *out_image_score = max_score;
    return 0;
}

PCN_API const char* pcn_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
