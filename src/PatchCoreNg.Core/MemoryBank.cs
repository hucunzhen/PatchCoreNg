namespace PatchCoreNg;

public sealed class MemoryBank : IDisposable
{
    public float[][] Embeddings { get; }
    private readonly IMemoryBankSearcher? _searcher;
    private readonly KnnsSearchOptions _options;
    private readonly NativeKnns.NativeMemoryBankHandle? _nativeBank;
    private readonly bool _useNative;

    public bool UsesNativeAcceleration => _useNative;

    public MemoryBank(float[][] embeddings, KnnsSearchOptions? options = null)
    {
        if (embeddings.Length == 0)
            throw new ArgumentException("Memory Bank 不能为空。");

        Embeddings = embeddings;
        _options = options ?? new KnnsSearchOptions();
        _options.Validate();

        if (NativeKnns.TryCreateBank(embeddings, _options, out var nativeBank) && nativeBank is not null)
        {
            _nativeBank = nativeBank;
            _useNative = true;
            _searcher = null;
            return;
        }

        _searcher = MemoryBankSearcherFactory.Create(embeddings, _options);
        _useNative = false;
    }

    public (float[] Distances, float ImageScore, float MaxScore) Score(float[][] queryPatches, int numNeighbors)
    {
        if (queryPatches.Length == 0)
            return ([], 0f, 0f);

        if (_useNative)
            throw new NotSupportedException("Native Memory Bank 请使用 ScoreFeatureMap 接口。");

        var distances = new float[queryPatches.Length];
        var parallelism = KnnsSearchOptions.ResolvePatchParallelism(_options.PatchScoreParallelism);

        Parallel.For(0, queryPatches.Length, new ParallelOptions
        {
            MaxDegreeOfParallelism = parallelism,
        }, i =>
        {
            var nn = _searcher!.NearestNeighborDistances(queryPatches[i], numNeighbors, _options);
            distances[i] = nn.Length > 0 ? nn.Average() : 0f;
        });

        var maxScore = distances.Max();
        return (distances, maxScore, maxScore);
    }

    public (float[] Distances, float ImageScore) ScoreFeatureMap(
        FeatureMap featureMap,
        int patchSize,
        int numNeighbors)
    {
        if (_useNative && _nativeBank is not null)
        {
            var patchCount = featureMap.Height * featureMap.Width;
            var distances = new float[patchCount];
            var parallelism = KnnsSearchOptions.ResolvePatchParallelism(_options.PatchScoreParallelism);
            NativeKnns.TryAggregateAndScore(
                _nativeBank,
                featureMap,
                patchSize,
                numNeighbors,
                parallelism,
                distances,
                out var imageScore);
            return (distances, imageScore);
        }

        var patches = LocalAggregator.Aggregate(featureMap, patchSize);
        var (legacyDistances, legacyScore, _) = Score(patches, numNeighbors);
        return (legacyDistances, legacyScore);
    }

    public void Dispose()
    {
        _nativeBank?.Dispose();
    }
}
