namespace PatchCoreNg;

public sealed class KnnsSearchOptions
{
    public DistanceMetric DistanceMetric { get; init; } = DistanceMetric.SquaredEuclidean;
    public bool UseSimdDistance { get; init; } = true;
    public int PatchScoreParallelism { get; init; }
    public int FeatureMapDownscale { get; init; } = 1;
    public bool UseApproximateNearestNeighbors { get; init; }
    public int AnnClusterCount { get; init; } = 32;
    public int AnnProbeClusters { get; init; } = 4;

    public static KnnsSearchOptions FromConfig(PatchCoreConfig config) => new()
    {
        DistanceMetric = config.DistanceMetric,
        UseSimdDistance = config.UseSimdDistance,
        PatchScoreParallelism = config.PatchScoreParallelism,
        FeatureMapDownscale = config.FeatureMapDownscale,
        UseApproximateNearestNeighbors = config.UseApproximateNearestNeighbors,
        AnnClusterCount = config.AnnClusterCount,
        AnnProbeClusters = config.AnnProbeClusters,
    };

    public static KnnsSearchOptions FromModel(PatchCoreModel model, PatchCoreConfig? userConfig = null)
    {
        var options = userConfig is null ? new KnnsSearchOptions() : FromConfig(userConfig);
        return new KnnsSearchOptions
        {
            DistanceMetric = model.DistanceMetric,
            UseSimdDistance = userConfig?.UseSimdDistance ?? model.UseSimdDistance,
            PatchScoreParallelism = userConfig?.PatchScoreParallelism ?? model.PatchScoreParallelism,
            FeatureMapDownscale = userConfig?.FeatureMapDownscale ?? model.FeatureMapDownscale,
            UseApproximateNearestNeighbors = userConfig?.UseApproximateNearestNeighbors ?? model.UseApproximateNearestNeighbors,
            AnnClusterCount = userConfig?.AnnClusterCount ?? model.AnnClusterCount,
            AnnProbeClusters = userConfig?.AnnProbeClusters ?? model.AnnProbeClusters,
        };
    }

    public static int ResolvePatchParallelism(int configured) =>
        configured > 0 ? configured : FeaturePipeline.ResolvePreprocessParallelism(0);

    public void Validate()
    {
        if (FeatureMapDownscale < 1 || FeatureMapDownscale > 8)
            throw new ArgumentOutOfRangeException(nameof(FeatureMapDownscale), "特征图降采样倍数须在 1~8。");

        if (AnnClusterCount < 2)
            throw new ArgumentOutOfRangeException(nameof(AnnClusterCount), "ANN 聚类数至少为 2。");

        if (AnnProbeClusters < 1)
            throw new ArgumentOutOfRangeException(nameof(AnnProbeClusters), "ANN 探测簇数至少为 1。");
    }
}
