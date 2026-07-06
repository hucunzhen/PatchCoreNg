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



        return FindBestThreshold(okScores, ngScores, numNeighbors);

    }



    public static TuningMetrics TuneNeighbors(

        PatchCoreModel model,

        PatchCoreConfig config,

        IReadOnlyList<string> okPaths,

        IReadOnlyList<string> ngPaths,

        IEnumerable<int>? neighborCandidates = null,

        IProgress<string>? progress = null)

    {

        var log = new StepProgress(progress);

        var candidates = neighborCandidates?.Distinct().OrderBy(x => x).ToArray()

            ?? DefaultNeighborCandidates;



        log.Begin("调参-kNN搜索", $"候选 k={string.Join(",", candidates)} | OK={okPaths.Count} NG={ngPaths.Count}");

        var searchWatch = System.Diagnostics.Stopwatch.StartNew();



        TuningMetrics? best = null;

        foreach (var neighbors in candidates)

        {

            var metrics = log.Run(

                $"调参-k={neighbors}",

                () =>

                {

                    var okScores = ScoreImages(model, config, neighbors, okPaths, log, $"调参-k{neighbors}-OK");

                    var ngScores = ScoreImages(model, config, neighbors, ngPaths, log, $"调参-k{neighbors}-NG");

                    return FindBestThreshold(okScores, ngScores, neighbors);

                },

                $"OK={okPaths.Count}, NG={ngPaths.Count}",

                result => $"阈值={result.Threshold:F4}, F1={result.F1:P1}");



            if (best is null || metrics.F1 > best.F1)

                best = metrics;

        }



        searchWatch.Stop();

        log.End(

            "调参-kNN搜索",

            searchWatch.Elapsed,

            best is null

                ? "无结果"

                : $"最佳 k={best.NumNeighbors}, 阈值={best.Threshold:F4}, F1={best.F1:P1}");



        return best ?? throw new InvalidOperationException("调参失败。");

    }



    public static IReadOnlyList<float> ScoreImages(

        PatchCoreModel model,

        PatchCoreConfig config,

        int numNeighbors,

        IEnumerable<string> imagePaths,

        StepProgress? log = null,

        string prefix = "调参打分")

    {

        var paths = imagePaths.ToList();

        using var extractor = new FeatureExtractor(

            config.BackboneOnnxPath,

            config.ImageSize,

            config.UseGpu,

            config.GpuDeviceId);

        var memoryBank = new MemoryBank(model.MemoryBank, KnnsSearchOptions.FromModel(model, config));

        return FeaturePipeline.ScoreImages(extractor, memoryBank, paths, config, numNeighbors, log, prefix);

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

        float threshold,

        IProgress<string>? progress = null)

    {

        var log = new StepProgress(progress);

        var okScores = log.Run(

            "测试-OK打分",

            () => ScoreImages(model, config, numNeighbors, okPaths, log, "测试-OK").ToList(),

            $"{okPaths.Count} 张",

            scores => $"{scores.Count} 个分数");

        var ngScores = log.Run(

            "测试-NG打分",

            () => ScoreImages(model, config, numNeighbors, ngPaths, log, "测试-NG").ToList(),

            $"{ngPaths.Count} 张",

            scores => $"{scores.Count} 个分数");



        return log.Run(

            "测试-指标计算",

            () => EvaluateAtScores(okScores, ngScores, threshold, numNeighbors),

            $"阈值={threshold:F4}, k={numNeighbors}",

            metrics =>

                $"F1={metrics.F1:P1}, Acc={metrics.Accuracy:P1}, TP={metrics.TruePositive}, TN={metrics.TrueNegative}, FP={metrics.FalsePositive}, FN={metrics.FalseNegative}");

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


