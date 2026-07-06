namespace PatchCoreNg;

public sealed class MemoryBank
{
    public float[][] Embeddings { get; }
    private readonly IMemoryBankSearcher _searcher;
    private readonly KnnsSearchOptions _options;

    public MemoryBank(float[][] embeddings, KnnsSearchOptions? options = null)
    {
        if (embeddings.Length == 0)
            throw new ArgumentException("Memory Bank 不能为空。");

        Embeddings = embeddings;
        _options = options ?? new KnnsSearchOptions();
        _options.Validate();
        _searcher = MemoryBankSearcherFactory.Create(embeddings, _options);
    }

    public (float[] Distances, float ImageScore, float MaxScore) Score(float[][] queryPatches, int numNeighbors)
    {
        if (queryPatches.Length == 0)
            return ([], 0f, 0f);

        var distances = new float[queryPatches.Length];
        var parallelism = KnnsSearchOptions.ResolvePatchParallelism(_options.PatchScoreParallelism);

        Parallel.For(0, queryPatches.Length, new ParallelOptions
        {
            MaxDegreeOfParallelism = parallelism,
        }, i =>
        {
            var nn = _searcher.NearestNeighborDistances(queryPatches[i], numNeighbors, _options);
            distances[i] = nn.Length > 0 ? nn.Average() : 0f;
        });

        var maxScore = distances.Max();
        return (distances, maxScore, maxScore);
    }
}
