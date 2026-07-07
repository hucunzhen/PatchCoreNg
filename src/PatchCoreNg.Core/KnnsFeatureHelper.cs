namespace PatchCoreNg;

public static class KnnsFeatureHelper
{
    public static FeatureMap PrepareFeatureMap(FeatureMap map, KnnsSearchOptions options)
    {
        options.Validate();
        return FeatureMapDownscaler.Downscale(map, options.FeatureMapDownscale);
    }

    public static float[][] AggregatePatches(FeatureMap map, int patchSize) =>
        LocalAggregator.Aggregate(map, patchSize);

    public static (float[] PatchDistances, float ImageScore) ScorePatches(
        MemoryBank memoryBank,
        float[][] patches,
        int numNeighbors)
    {
        var (distances, imageScore, _) = memoryBank.Score(patches, numNeighbors);
        return (distances, imageScore);
    }

    public static FeatureMapScoreDetail ScoreFeatureMap(
        MemoryBank memoryBank,
        FeatureMap featureMap,
        int patchSize,
        int numNeighbors,
        KnnsSearchOptions options)
    {
        var prepared = PrepareFeatureMap(featureMap, options);
        if (prepared.Height == 0 || prepared.Width == 0)
        {
            throw new InvalidOperationException(
                $"特征图为空 ({prepared.Channels}x{prepared.Height}x{prepared.Width})。");
        }

        var (distances, imageScore) = memoryBank.ScoreFeatureMap(prepared, patchSize, numNeighbors);
        return new FeatureMapScoreDetail(imageScore, distances, prepared);
    }
}

public readonly record struct FeatureMapScoreDetail(
    float ImageScore,
    float[] PatchDistances,
    FeatureMap ScoredFeatureMap)
{
    public FeatureMapScoreDetail(float imageScore, float[] patchDistances)
        : this(imageScore, patchDistances, new FeatureMap([], 0, 0, 0))
    {
    }
}
