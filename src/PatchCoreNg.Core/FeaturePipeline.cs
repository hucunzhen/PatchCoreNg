using System.Diagnostics;

namespace PatchCoreNg;

internal static class FeaturePipeline
{
    public static int ResolvePreprocessParallelism(int configured)
    {
        if (configured > 0)
            return configured;

        return Math.Clamp(Environment.ProcessorCount, 2, 16);
    }

    public static int ResolveBatchSize(int configured)
    {
        return Math.Clamp(configured <= 0 ? 8 : configured, 1, 64);
    }

    public static float[][] PreprocessImages(
        IReadOnlyList<string> imagePaths,
        int imageSize,
        int parallelism,
        StepProgress? log = null,
        string step = "图像预处理")
    {
        log?.Begin(step, $"{imagePaths.Count} 张, 并行={ResolvePreprocessParallelism(parallelism)}");
        var watch = Stopwatch.StartNew();
        var tensors = new float[imagePaths.Count][];
        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = ResolvePreprocessParallelism(parallelism),
        };
        var lastReport = 0;

        Parallel.For(0, imagePaths.Count, options, i =>
        {
            tensors[i] = ImagePreprocessor.LoadAndPreprocess(imagePaths[i], imageSize);
            var done = i + 1;
            if (log != null && done - lastReport >= 64)
            {
                Interlocked.Exchange(ref lastReport, done);
                log.Info(step, $"进度 {done}/{imagePaths.Count} | 已用 {StepProgress.FormatElapsed(watch.Elapsed)}");
            }
        });

        watch.Stop();
        log?.End(step, watch.Elapsed, $"{imagePaths.Count} 张");
        return tensors;
    }

    public static List<FeatureMap> ExtractFeatureMaps(
        FeatureExtractor extractor,
        float[][] tensors,
        int batchSize,
        StepProgress? log = null,
        string step = "特征提取")
    {
        batchSize = ResolveBatchSize(batchSize);
        log?.Begin(step, $"{tensors.Length} 张, batch={batchSize}, 设备={extractor.ExecutionProvider}");
        var watch = Stopwatch.StartNew();
        var maps = new List<FeatureMap>(tensors.Length);

        for (var offset = 0; offset < tensors.Length; offset += batchSize)
        {
            var count = Math.Min(batchSize, tensors.Length - offset);
            var batch = new float[count][];
            Array.Copy(tensors, offset, batch, 0, count);
            maps.AddRange(extractor.ExtractBatch(batch));

            var done = Math.Min(offset + count, tensors.Length);
            if (log != null && (done == tensors.Length || done % (batchSize * 4) == 0))
                log.Info(step, $"进度 {done}/{tensors.Length} | 已用 {StepProgress.FormatElapsed(watch.Elapsed)}");
        }

        watch.Stop();
        log?.End(step, watch.Elapsed, $"{maps.Count} 张特征图");
        return maps;
    }

    public static List<float[]> ExtractPatches(
        IReadOnlyList<FeatureMap> featureMaps,
        int patchSize,
        int parallelism,
        StepProgress? log = null,
        string step = "Patch聚合")
    {
        log?.Begin(step, $"{featureMaps.Count} 张特征图, patch={patchSize}");
        var watch = Stopwatch.StartNew();
        var patchesPerImage = new float[featureMaps.Count][][];
        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = ResolvePreprocessParallelism(parallelism),
        };

        Parallel.For(0, featureMaps.Count, options, i =>
        {
            patchesPerImage[i] = LocalAggregator.Aggregate(featureMaps[i], patchSize);
        });

        var allPatches = new List<float[]>();
        foreach (var patches in patchesPerImage)
            allPatches.AddRange(patches);

        watch.Stop();
        log?.End(step, watch.Elapsed, $"{allPatches.Count} 个 patch");
        return allPatches;
    }

    public static List<float> ScoreFeatureMaps(
        MemoryBank memoryBank,
        IReadOnlyList<FeatureMap> featureMaps,
        int patchSize,
        int numNeighbors,
        int parallelism,
        StepProgress? log = null,
        string step = "kNN打分")
    {
        log?.Begin(step, $"{featureMaps.Count} 张, k={numNeighbors}");
        var watch = Stopwatch.StartNew();
        var scores = new float[featureMaps.Count];

        Parallel.For(0, featureMaps.Count, new ParallelOptions
        {
            MaxDegreeOfParallelism = ResolvePreprocessParallelism(parallelism),
        }, i =>
        {
            var patches = LocalAggregator.Aggregate(featureMaps[i], patchSize);
            scores[i] = memoryBank.Score(patches, numNeighbors).ImageScore;
        });

        watch.Stop();
        log?.End(step, watch.Elapsed, $"{scores.Length} 个分数");
        return scores.ToList();
    }

    public static List<float> ScoreImages(
        FeatureExtractor extractor,
        MemoryBank memoryBank,
        IReadOnlyList<string> imagePaths,
        PatchCoreConfig config,
        int numNeighbors,
        StepProgress? log = null,
        string prefix = "调参打分")
    {
        if (imagePaths.Count == 0)
            return [];

        var tensors = PreprocessImages(
            imagePaths,
            config.ImageSize,
            config.PreprocessParallelism,
            log,
            $"{prefix}-预处理");
        var featureMaps = ExtractFeatureMaps(
            extractor,
            tensors,
            config.InferenceBatchSize,
            log,
            $"{prefix}-特征提取");
        return ScoreFeatureMaps(
            memoryBank,
            featureMaps,
            config.PatchSize,
            numNeighbors,
            config.PreprocessParallelism,
            log,
            $"{prefix}-kNN");
    }
}
