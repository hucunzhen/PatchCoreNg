namespace PatchCoreNg;

public sealed class TuningMetrics
{
    public required float Threshold { get; init; }
    public required int NumNeighbors { get; init; }
    public required float Accuracy { get; init; }
    public required float Precision { get; init; }
    public required float Recall { get; init; }
    public required float F1 { get; init; }
    public required int TruePositive { get; init; }
    public required int TrueNegative { get; init; }
    public required int FalsePositive { get; init; }
    public required int FalseNegative { get; init; }
    public required int OkCount { get; init; }
    public required int NgCount { get; init; }
}

public static class ParameterTuner
{
    public static readonly int[] DefaultNeighborCandidates = [1, 3, 5, 9, 15];

    public static TuningMetrics Tune(
        PatchCoreModel model,
        PatchCoreConfig config,
        IReadOnlyList<float> okScores,
        IReadOnlyList<float> ngScores,
        int numNeighbors)
    {
        if (okScores.Count == 0 || ngScores.Count == 0)
            throw new InvalidOperationException("调参需要至少 1 张 OK 样本和 1 张 NG 样本。");

        var best = FindBestThreshold(okScores, ngScores, numNeighbors);
        return best;
    }

    public static TuningMetrics TuneNeighbors(
        PatchCoreModel model,
        PatchCoreConfig config,
        IReadOnlyList<string> okPaths,
        IReadOnlyList<string> ngPaths,
        IEnumerable<int>? neighborCandidates = null,
        IProgress<string>? progress = null)
    {
        var candidates = neighborCandidates?.Distinct().OrderBy(x => x).ToArray()
            ?? DefaultNeighborCandidates;

        TuningMetrics? best = null;
        foreach (var neighbors in candidates)
        {
            progress?.Report($"搜索 kNN 邻居数: {neighbors}");
            var okScores = ScoreImages(model, config, neighbors, okPaths);
            var ngScores = ScoreImages(model, config, neighbors, ngPaths);
            var metrics = FindBestThreshold(okScores, ngScores, neighbors);
            if (best is null || metrics.F1 > best.F1)
                best = metrics;
        }

        return best ?? throw new InvalidOperationException("调参失败。");
    }

    public static IReadOnlyList<float> ScoreImages(
        PatchCoreModel model,
        PatchCoreConfig config,
        int numNeighbors,
        IEnumerable<string> imagePaths)
    {
        using var extractor = new FeatureExtractor(config.BackboneOnnxPath, config.ImageSize);
        var memoryBank = new MemoryBank(model.MemoryBank);
        var scores = new List<float>();

        foreach (var path in imagePaths)
        {
            var tensor = ImagePreprocessor.LoadAndPreprocess(path, config.ImageSize);
            var featureMap = extractor.Extract(tensor);
            var patches = LocalAggregator.Aggregate(featureMap, config.PatchSize);
            scores.Add(memoryBank.Score(patches, numNeighbors).ImageScore);
        }

        return scores;
    }

    public static TuningMetrics FindBestThreshold(
        IReadOnlyList<float> okScores,
        IReadOnlyList<float> ngScores,
        int numNeighbors)
    {
        var candidates = okScores.Concat(ngScores)
            .Distinct()
            .OrderBy(x => x)
            .ToList();

        if (candidates.Count == 1)
            candidates.Add(candidates[0] + 1e-4f);

        TuningMetrics? best = null;
        for (var i = 0; i < candidates.Count - 1; i++)
        {
            var threshold = (candidates[i] + candidates[i + 1]) / 2f;
            var metrics = Evaluate(okScores, ngScores, threshold, numNeighbors);
            if (best is null || metrics.F1 > best.F1)
                best = metrics;
        }

        // 也尝试边界阈值
        var minScore = candidates.First() - 1e-4f;
        var maxScore = candidates.Last() + 1e-4f;
        foreach (var threshold in new[] { minScore, maxScore })
        {
            var metrics = Evaluate(okScores, ngScores, threshold, numNeighbors);
            if (best is null || metrics.F1 > best.F1)
                best = metrics;
        }

        return best ?? throw new InvalidOperationException("无法计算调参指标。");
    }

    public static TuningMetrics EvaluateAtScores(
        IReadOnlyList<float> okScores,
        IReadOnlyList<float> ngScores,
        float threshold,
        int numNeighbors)
    {
        if (okScores.Count == 0 || ngScores.Count == 0)
            throw new InvalidOperationException("评估需要至少 1 张 OK 样本和 1 张 NG 样本。");

        return Evaluate(okScores, ngScores, threshold, numNeighbors);
    }

    public static TuningMetrics EvaluateAtThreshold(
        PatchCoreModel model,
        PatchCoreConfig config,
        int numNeighbors,
        IReadOnlyList<string> okPaths,
        IReadOnlyList<string> ngPaths,
        float threshold)
    {
        var okScores = ScoreImages(model, config, numNeighbors, okPaths);
        var ngScores = ScoreImages(model, config, numNeighbors, ngPaths);
        return EvaluateAtScores(okScores, ngScores, threshold, numNeighbors);
    }

    private static TuningMetrics Evaluate(
        IReadOnlyList<float> okScores,
        IReadOnlyList<float> ngScores,
        float threshold,
        int numNeighbors)
    {
        var tp = ngScores.Count(s => s > threshold);
        var fn = ngScores.Count - tp;
        var tn = okScores.Count(s => s <= threshold);
        var fp = okScores.Count - tn;

        var total = tp + tn + fp + fn;
        var accuracy = total == 0 ? 0 : (tp + tn) / (float)total;
        var precision = tp + fp == 0 ? 0 : tp / (float)(tp + fp);
        var recall = tp + fn == 0 ? 0 : tp / (float)(tp + fn);
        var f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);

        return new TuningMetrics
        {
            Threshold = threshold,
            NumNeighbors = numNeighbors,
            Accuracy = accuracy,
            Precision = precision,
            Recall = recall,
            F1 = f1,
            TruePositive = tp,
            TrueNegative = tn,
            FalsePositive = fp,
            FalseNegative = fn,
            OkCount = okScores.Count,
            NgCount = ngScores.Count
        };
    }
}
