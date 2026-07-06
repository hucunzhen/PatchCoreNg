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
        var patches = AggregatePatches(prepared, patchSize);
        if (patches.Length == 0)
        {
            throw new InvalidOperationException(
                $"特征图为空 ({prepared.Channels}x{prepared.Height}x{prepared.Width})。");
        }

        var (distances, imageScore) = ScorePatches(memoryBank, patches, numNeighbors);
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
