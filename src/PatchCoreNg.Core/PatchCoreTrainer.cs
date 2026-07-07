namespace PatchCoreNg;

public sealed class PatchCoreTrainer : IDisposable
{
    private readonly PatchCoreConfig _config;
    private readonly FeatureExtractor _extractor;

    public PatchCoreTrainer(PatchCoreConfig config)
    {
        _config = config;
        _extractor = new FeatureExtractor(
            config.BackboneOnnxPath,
            config.ImageSize,
            config.UseGpu,
            config.GpuDeviceId);
    }

    public PatchCoreModel Train(string trainDataPath, IProgress<string>? progress = null)
    {
        var imagePaths = ImagePreprocessor.EnumerateImages(trainDataPath).ToList();
        if (imagePaths.Count == 0)
            throw new InvalidOperationException($"训练目录中没有图像: {trainDataPath}");

        return Train(imagePaths, progress);
    }

    public PatchCoreModel Train(IReadOnlyList<string> imagePaths, IProgress<string>? progress = null)
    {
        if (imagePaths.Count == 0)
            throw new InvalidOperationException("训练样本列表为空。");

        var log = new StepProgress(progress);
        log.Info(
            "训练",
            $"{imagePaths.Count} 张 OK 样本 | 设备={_extractor.ExecutionProvider} | " +
            $"batch={FeaturePipeline.ResolveBatchSize(_config.InferenceBatchSize)} | " +
            $"预处理并行={FeaturePipeline.ResolvePreprocessParallelism(_config.PreprocessParallelism)}");

        var tensors = FeaturePipeline.PreprocessImages(
            imagePaths, _config.ImageSize, _config.PreprocessParallelism, log, "训练-预处理");
        var featureMaps = FeaturePipeline.ExtractFeatureMaps(
            _extractor, tensors, _config.InferenceBatchSize, log, "训练-特征提取");
        EmbedDimension.EnsureFeatureMapMatches(
            featureMaps[0],
            _config.TargetEmbedDimension,
            _config.BackboneOnnxPath,
            _config.BackboneId,
            _config.ImageSize);
        log.Info("训练", $"特征维度={featureMaps[0].Channels} (TargetEmbedDimension={_config.TargetEmbedDimension})");
        var allPatches = FeaturePipeline.ExtractPatches(
            featureMaps, _config.PatchSize, _config.PreprocessParallelism, log, "训练-Patch聚合");
        var coreset = CoresetSampler.Sample(allPatches.ToArray(), _config.CoresetRatio, log, "训练-Coreset采样");

        var referenceMap = featureMaps[0];
        var knnsOptions = KnnsSearchOptions.FromConfig(_config);
        using var memoryBank = new MemoryBank(coreset, knnsOptions);
        var trainScores = FeaturePipeline.ScoreFeatureMaps(
            memoryBank,
            featureMaps,
            _config.PatchSize,
            _config.NumNeighbors,
            knnsOptions,
            _config.PreprocessParallelism,
            log,
            "训练-打分");

        var sortedScores = trainScores.OrderBy(x => x).ToArray();
        var p95Index = (int)Math.Clamp(
            Math.Round(sortedScores.Length * 0.95) - 1,
            0,
            sortedScores.Length - 1);
        var threshold = sortedScores[p95Index];
        log.Info("训练-阈值估计", $"P95 阈值={threshold:F4}");

        return PatchCoreModel.Create(_config, referenceMap, coreset, threshold);
    }

    public void Dispose() => _extractor.Dispose();
}
