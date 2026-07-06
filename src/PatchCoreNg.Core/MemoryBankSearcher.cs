namespace PatchCoreNg;

internal interface IMemoryBankSearcher
{
    float[] NearestNeighborDistances(float[] query, int k, KnnsSearchOptions options);
}

internal static class TopKDistance
{
    public static float[] AverageTopK(float[] topK, int validCount, DistanceMetric metric)
    {
        if (validCount <= 0)
            return [];

        var values = new float[validCount];
        Array.Copy(topK, values, validCount);

        if (metric == DistanceMetric.SquaredEuclidean)
        {
            for (var i = 0; i < values.Length; i++)
                values[i] = MathF.Sqrt(values[i]);
        }

        return values;
    }

    public static void InsertTopKSquared(float[] topK, float squaredDistance)
    {
        var maxIdx = 0;
        for (var i = 1; i < topK.Length; i++)
        {
            if (topK[i] > topK[maxIdx])
                maxIdx = i;
        }

        if (squaredDistance < topK[maxIdx])
            topK[maxIdx] = squaredDistance;
    }

    public static int CountValid(float[] topK) =>
        topK.Count(d => d < float.MaxValue);
}

internal sealed class BruteForceMemoryBankSearcher(float[][] embeddings) : IMemoryBankSearcher
{
    public float[] NearestNeighborDistances(float[] query, int k, KnnsSearchOptions options)
    {
        var topK = new float[k];
        Array.Fill(topK, float.MaxValue);

        foreach (var candidate in embeddings)
        {
            var squared = VectorDistance.SquaredDistance(
                query,
                candidate,
                options.UseSimdDistance);
            TopKDistance.InsertTopKSquared(topK, squared);
        }

        var valid = TopKDistance.CountValid(topK);
        var averaged = TopKDistance.AverageTopK(topK, valid, options.DistanceMetric);
        return averaged.Length > 0 ? averaged : [];
    }
}

internal sealed class ClusterAnnMemoryBankSearcher : IMemoryBankSearcher
{
    private readonly float[][] _embeddings;
    private readonly float[][] _centroids;
    private readonly List<int>[] _clusters;

    public ClusterAnnMemoryBankSearcher(float[][] embeddings, int clusterCount, int seed = 0)
    {
        _embeddings = embeddings;
        clusterCount = Math.Clamp(clusterCount, 2, Math.Max(2, embeddings.Length));
        (_centroids, _clusters) = BuildClusters(embeddings, clusterCount, seed);
    }

    public float[] NearestNeighborDistances(float[] query, int k, KnnsSearchOptions options)
    {
        var probeCount = Math.Clamp(options.AnnProbeClusters, 1, _centroids.Length);
        var probes = SelectProbeCentroids(query, probeCount, options);
        var visited = new HashSet<int>();
        var topK = new float[k];
        Array.Fill(topK, float.MaxValue);

        foreach (var clusterIdx in probes)
        {
            foreach (var embeddingIdx in _clusters[clusterIdx])
            {
                if (!visited.Add(embeddingIdx))
                    continue;

                var squared = VectorDistance.SquaredDistance(
                    query,
                    _embeddings[embeddingIdx],
                    options.UseSimdDistance);
                TopKDistance.InsertTopKSquared(topK, squared);
            }
        }

        var valid = TopKDistance.CountValid(topK);
        if (valid < k)
            return new BruteForceMemoryBankSearcher(_embeddings).NearestNeighborDistances(query, k, options);

        var averaged = TopKDistance.AverageTopK(topK, valid, options.DistanceMetric);
        return averaged.Length > 0 ? averaged : [];
    }

    private int[] SelectProbeCentroids(float[] query, int probeCount, KnnsSearchOptions options)
    {
        var ranked = new (int Index, float Distance)[_centroids.Length];
        for (var i = 0; i < _centroids.Length; i++)
        {
            ranked[i] = (
                i,
                VectorDistance.SquaredDistance(query, _centroids[i], options.UseSimdDistance));
        }

        return ranked
            .OrderBy(x => x.Distance)
            .Take(probeCount)
            .Select(x => x.Index)
            .ToArray();
    }

    private static (float[][] Centroids, List<int>[] Clusters) BuildClusters(
        float[][] embeddings,
        int clusterCount,
        int seed)
    {
        var dim = embeddings[0].Length;
        var rng = new Random(seed);
        var centroids = new float[clusterCount][];
        var used = new HashSet<int>();

        for (var i = 0; i < clusterCount; i++)
        {
            var pick = rng.Next(embeddings.Length);
            while (!used.Add(pick))
                pick = rng.Next(embeddings.Length);

            centroids[i] = embeddings[pick].ToArray();
        }

        var clusters = Enumerable.Range(0, clusterCount).Select(_ => new List<int>()).ToArray();

        for (var iter = 0; iter < 8; iter++)
        {
            foreach (var cluster in clusters)
                cluster.Clear();

            for (var i = 0; i < embeddings.Length; i++)
            {
                var best = 0;
                var bestDist = float.MaxValue;
                for (var c = 0; c < clusterCount; c++)
                {
                    var dist = VectorDistance.SquaredDistance(
                        embeddings[i],
                        centroids[c],
                        useSimd: true);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        best = c;
                    }
                }

                clusters[best].Add(i);
            }

            for (var c = 0; c < clusterCount; c++)
            {
                if (clusters[c].Count == 0)
                    continue;

                var centroid = new float[dim];
                foreach (var idx in clusters[c])
                {
                    for (var d = 0; d < dim; d++)
                        centroid[d] += embeddings[idx][d];
                }

                for (var d = 0; d < dim; d++)
                    centroid[d] /= clusters[c].Count;
                centroids[c] = centroid;
            }
        }

        return (centroids, clusters);
    }
}

internal static class MemoryBankSearcherFactory
{
    public static IMemoryBankSearcher Create(float[][] embeddings, KnnsSearchOptions options)
    {
        options.Validate();

        if (options.UseApproximateNearestNeighbors && embeddings.Length >= options.AnnClusterCount)
        {
            return new ClusterAnnMemoryBankSearcher(
                embeddings,
                options.AnnClusterCount,
                seed: 0);
        }

        return new BruteForceMemoryBankSearcher(embeddings);
    }
}
