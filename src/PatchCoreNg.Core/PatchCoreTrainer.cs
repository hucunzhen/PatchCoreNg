namespace PatchCoreNg;

public sealed class PatchCoreTrainer : IDisposable
{
    private readonly PatchCoreConfig _config;
    private readonly FeatureExtractor _extractor;

    public PatchCoreTrainer(PatchCoreConfig config)
    {
        _config = config;
        _extractor = new FeatureExtractor(config.BackboneOnnxPath, config.ImageSize);
    }

    public PatchCoreModel Train(string trainDataPath, IProgress<string>? progress = null)
    {
        var imagePaths = ImagePreprocessor.EnumerateImages(trainDataPath).ToList();
        if (imagePaths.Count == 0)
            throw new InvalidOperationException($"训练目录中没有图像: {trainDataPath}");

        progress?.Report($"发现 {imagePaths.Count} 张正常样本，开始提取特征...");

        var allPatches = new List<float[]>();
        FeatureMap referenceMap = default;
        var hasReference = false;

        for (var i = 0; i < imagePaths.Count; i++)
        {
            var path = imagePaths[i];
            progress?.Report($"[{i + 1}/{imagePaths.Count}] {Path.GetFileName(path)}");

            var tensor = ImagePreprocessor.LoadAndPreprocess(path, _config.ImageSize);
            var featureMap = _extractor.Extract(tensor);
            if (!hasReference)
            {
                referenceMap = featureMap;
                hasReference = true;
            }

            var patches = LocalAggregator.Aggregate(featureMap, _config.PatchSize);
            allPatches.AddRange(patches);
        }

        progress?.Report($"共 {allPatches.Count} 个 patch，coreset 采样比例 {_config.CoresetRatio:P0}...");
        var coreset = CoresetSampler.Sample(allPatches.ToArray(), _config.CoresetRatio, progress);
        progress?.Report($"Memory Bank 大小: {coreset.Length}");

        if (!hasReference)
            throw new InvalidOperationException("未能从训练数据中提取特征。");

        var memoryBank = new MemoryBank(coreset);
        var trainScores = new List<float>();
        foreach (var path in imagePaths)
        {
            var tensor = ImagePreprocessor.LoadAndPreprocess(path, _config.ImageSize);
            var featureMap = _extractor.Extract(tensor);
            var patches = LocalAggregator.Aggregate(featureMap, _config.PatchSize);
            trainScores.Add(memoryBank.Score(patches, _config.NumNeighbors).ImageScore);
        }

        trainScores.Sort();
        var thresholdIndex = (int)Math.Clamp(
            Math.Round(trainScores.Count * 0.95) - 1,
            0,
            trainScores.Count - 1);
        var threshold = trainScores[thresholdIndex];
        progress?.Report($"自动阈值 (P95): {threshold:F4}");

        return PatchCoreModel.Create(_config, referenceMap, coreset, threshold);
    }

    public void Dispose() => _extractor.Dispose();
}
