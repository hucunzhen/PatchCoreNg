#pragma once

#ifdef _WIN32
#  ifdef PCN_EXPORTS
#    define PCN_API __declspec(dllexport)
#  else
#    define PCN_API __declspec(dllimport)
#  endif
#else
#  define PCN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum PcnDistanceMetric {
    PCN_EUCLIDEAN = 0,
    PCN_SQUARED_EUCLIDEAN = 1,
};

typedef struct PcnBankConfig {
    int use_ann;
    int ann_cluster_count;
    int ann_probe_clusters;
    int distance_metric;
    int use_simd;
} PcnBankConfig;

typedef struct PcnScoreParams {
    int patch_size;
    int num_neighbors;
    int patch_parallelism; /* 0 = use all cores */
} PcnScoreParams;

/* Opaque memory bank handle (embeddings + optional ANN index). */
typedef struct PcnMemoryBank PcnMemoryBank;

/*
 * Create bank from row-major embeddings [bank_count][dim].
 * Returns nullptr on failure.
 */
PCN_API PcnMemoryBank* pcn_bank_create(
    const float* embeddings,
    int bank_count,
    int dim,
    const PcnBankConfig* config);

PCN_API void pcn_bank_destroy(PcnMemoryBank* bank);

/*
 * Feature map layout: CHW, size = channels * height * width.
 * out_distances: height * width patch scores.
 * out_image_score: max patch score.
 * Returns 0 on success, negative error code otherwise.
 */
PCN_API int pcn_aggregate_and_score(
    const PcnMemoryBank* bank,
    const float* feature_chw,
    int channels,
    int height,
    int width,
    const PcnScoreParams* params,
    float* out_distances,
    float* out_image_score);

PCN_API const char* pcn_last_error(void);

#ifdef __cplusplus
}
#endif
