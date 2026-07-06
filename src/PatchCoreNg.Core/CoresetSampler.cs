using System.Diagnostics;

namespace PatchCoreNg;

public static class CoresetSampler
{
    private const int ReportIntervalMs = 400;

    /// <summary>
    /// 贪心 coreset 采样 (approx_greedy_coreset)。
    /// </summary>
    public static float[][] Sample(
        float[][] embeddings,
        double ratio,
        StepProgress? log = null,
        string step = "Coreset采样")
    {
        if (embeddings.Length == 0)
            return [];

        var n = embeddings.Length;
        var dim = embeddings[0].Length;
        var targetCount = Math.Max(1, (int)Math.Round(n * ratio));

        if (targetCount >= n)
        {
            log?.Info(step, $"保留全部 {n} 个 patch");
            return embeddings;
        }

        log?.Begin(step, $"目标 {targetCount}/{n} ({ratio:P0}), 维度 {dim}");

        var flat = new float[n * dim];
        for (var i = 0; i < n; i++)
            embeddings[i].AsSpan().CopyTo(flat.AsSpan(i * dim, dim));

        var minDistances = new float[n];
        Array.Fill(minDistances, float.MaxValue);

        var selectedIndices = new int[targetCount];
        var rng = new Random(0);
        var first = rng.Next(n);
        selectedIndices[0] = first;
        var selectedCount = 1;

        UpdateMinDistances(flat, dim, n, first, minDistances);

        var stopwatch = Stopwatch.StartNew();
        var lastReportMs = 0L;
        var lastReportedPercent = -1;

        while (selectedCount < targetCount)
        {
            var farthest = ArgMax(minDistances);
            selectedIndices[selectedCount] = farthest;
            selectedCount++;

            UpdateMinDistances(flat, dim, n, farthest, minDistances);

            var percent = (int)Math.Round(selectedCount * 100.0 / targetCount);
            var elapsedMs = stopwatch.ElapsedMilliseconds;
            var shouldReport = selectedCount == targetCount
                || percent > lastReportedPercent
                || elapsedMs - lastReportMs >= ReportIntervalMs;

            if (shouldReport && log != null)
            {
                lastReportMs = elapsedMs;
                lastReportedPercent = percent;
                var elapsed = stopwatch.Elapsed;
                var rate = selectedCount / Math.Max(elapsed.TotalSeconds, 0.001);
                var remaining = rate > 0 ? (targetCount - selectedCount) / rate : 0;
                log.Info(
                    step,
                    $"进度 {selectedCount}/{targetCount} ({percent}%) | " +
                    $"已用 {StepProgress.FormatElapsed(elapsed)} | 预计剩余 {remaining:F0}s");
            }
        }

        stopwatch.Stop();

        var coreset = new float[selectedCount][];
        for (var i = 0; i < selectedCount; i++)
        {
            var idx = selectedIndices[i];
            var vec = new float[dim];
            flat.AsSpan(idx * dim, dim).CopyTo(vec);
            coreset[i] = vec;
        }

        log?.End(step, stopwatch.Elapsed, $"{selectedCount} 个 patch");
        return coreset;
    }

    private static void UpdateMinDistances(
        float[] flat,
        int dim,
        int n,
        int centerIndex,
        float[] minDistances)
    {
        var centerOffset = centerIndex * dim;

        if (n >= 512)
        {
            Parallel.For(0, n, i =>
            {
                var d = SquaredDistanceFlat(flat, dim, i, centerOffset);
                if (d < minDistances[i])
                    minDistances[i] = d;
            });
            return;
        }

        for (var i = 0; i < n; i++)
        {
            var d = SquaredDistanceFlat(flat, dim, i, centerOffset);
            if (d < minDistances[i])
                minDistances[i] = d;
        }
    }

    private static float SquaredDistanceFlat(float[] flat, int dim, int pointIndex, int centerOffset)
    {
        var pointOffset = pointIndex * dim;
        var sum = 0f;
        for (var d = 0; d < dim; d++)
        {
            var diff = flat[pointOffset + d] - flat[centerOffset + d];
            sum += diff * diff;
        }

        return sum;
    }

    private static int ArgMax(float[] values)
    {
        var maxIdx = 0;
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] > values[maxIdx])
                maxIdx = i;
        }

        return maxIdx;
    }
}
