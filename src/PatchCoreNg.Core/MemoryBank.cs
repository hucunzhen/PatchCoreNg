namespace PatchCoreNg;

public sealed class MemoryBank
{
    public float[][] Embeddings { get; }

    public MemoryBank(float[][] embeddings)
    {
        Embeddings = embeddings;
    }

    public (float[] Distances, float ImageScore, float MaxScore) Score(float[][] queryPatches, int numNeighbors)
    {
        var distances = new float[queryPatches.Length];
        var maxScore = 0f;

        for (var i = 0; i < queryPatches.Length; i++)
        {
            var nnDistances = NearestNeighborDistances(queryPatches[i], numNeighbors);
            var score = nnDistances.Length > 0 ? nnDistances.Average() : 0f;
            distances[i] = score;
            if (score > maxScore)
                maxScore = score;
        }

        return (distances, maxScore, maxScore);
    }

    private float[] NearestNeighborDistances(float[] query, int k)
    {
        var topK = new float[k];
        Array.Fill(topK, float.MaxValue);

        foreach (var candidate in Embeddings)
        {
            var dist = MathF.Sqrt(SquaredDistance(query, candidate));
            InsertTopK(topK, dist);
        }

        return topK.Where(d => d < float.MaxValue).ToArray();
    }

    private static void InsertTopK(float[] topK, float value)
    {
        var maxIdx = 0;
        for (var i = 1; i < topK.Length; i++)
        {
            if (topK[i] > topK[maxIdx])
                maxIdx = i;
        }

        if (value < topK[maxIdx])
            topK[maxIdx] = value;
    }

    private static float SquaredDistance(float[] a, float[] b)
    {
        var sum = 0f;
        for (var i = 0; i < a.Length; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }

        return sum;
    }
}
