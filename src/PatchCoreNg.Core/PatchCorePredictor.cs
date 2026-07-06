using OpenCvSharp;

namespace PatchCoreNg;

public sealed class PredictionResult
{
    public required string ImagePath { get; init; }
    public required float AnomalyScore { get; init; }
    public required bool IsAnomaly { get; init; }
    public required string Label { get; init; }
    public string? HeatmapPath { get; init; }
}

public sealed class PatchCorePredictor : IDisposable
{
    private readonly PatchCoreConfig _config;
    private readonly FeatureExtractor _extractor;
    private readonly MemoryBank _memoryBank;
    private readonly PatchCoreModel _model;

    public string ExecutionProvider => _extractor.ExecutionProvider;
    internal FeatureExtractor Extractor => _extractor;
    internal MemoryBank MemoryBank => _memoryBank;

    public PatchCorePredictor(PatchCoreModel model, PatchCoreConfig? config = null)
    {
        _model = model;
        _config = config ?? new PatchCoreConfig
        {
            ImageSize = model.ImageSize,
            PatchSize = model.PatchSize,
            NumNeighbors = model.NumNeighbors,
            CoresetRatio = model.CoresetRatio,
            TargetEmbedDimension = model.TargetEmbedDimension,
            AnomalyThreshold = model.AnomalyThreshold
        };

        _extractor = new FeatureExtractor(
            _config.BackboneOnnxPath,
            _config.ImageSize,
            _config.UseGpu,
            _config.GpuDeviceId);
        _memoryBank = new MemoryBank(model.MemoryBank);
    }

    public PredictionResult Predict(string imagePath, string? outputDir = null)
    {
        var tensor = ImagePreprocessor.LoadAndPreprocess(imagePath, _config.ImageSize);
        var featureMap = _extractor.Extract(tensor);
        return PredictFromFeatureMap(imagePath, featureMap, outputDir);
    }

    public PredictionResult PredictFromFeatureMap(
        string imagePath,
        FeatureMap featureMap,
        string? outputDir = null)
    {
        var patches = LocalAggregator.Aggregate(featureMap, _config.PatchSize);
        if (patches.Length == 0)
        {
            throw new InvalidOperationException(
                $"特征图为空 ({featureMap.Channels}x{featureMap.Height}x{featureMap.Width})，" +
                "请检查 backbone ONNX 与训练时是否一致。");
        }

        var (patchDistances, imageScore, _) = _memoryBank.Score(patches, _config.NumNeighbors);
        return CreateResult(imagePath, featureMap, imageScore, patchDistances, outputDir);
    }

    public PredictionResult CreateResult(
        string imagePath,
        FeatureMap featureMap,
        FeatureMapScoreDetail scoreDetail,
        string? outputDir = null) =>
        CreateResult(imagePath, featureMap, scoreDetail.ImageScore, scoreDetail.PatchDistances, outputDir);

    public PredictionResult CreateResult(
        string imagePath,
        FeatureMap featureMap,
        float imageScore,
        float[] patchDistances,
        string? outputDir = null)
    {
        if (imageScore <= 0f && _memoryBank.Embeddings.Length == 0)
        {
            throw new InvalidOperationException(
                "Memory Bank 为空，推理分数恒为 0。请使用 patchcore_model.json 并重新训练。");
        }

        var isAnomaly = imageScore > GetThreshold();
        var label = isAnomaly ? "NG" : "OK";

        string? heatmapPath = null;
        if (_config.SaveHeatmap && !string.IsNullOrWhiteSpace(outputDir))
        {
            Directory.CreateDirectory(outputDir);
            heatmapPath = SaveHeatmap(imagePath, featureMap, patchDistances, outputDir, imageScore, label);
        }

        return new PredictionResult
        {
            ImagePath = imagePath,
            AnomalyScore = imageScore,
            IsAnomaly = isAnomaly,
            Label = label,
            HeatmapPath = heatmapPath
        };
    }

    private string SaveHeatmap(
        string imagePath,
        FeatureMap featureMap,
        float[] patchScores,
        string outputDir,
        float score,
        string label)
    {
        var threshold = GetThreshold();

        using var scoreMap = new Mat(featureMap.Height, featureMap.Width, MatType.CV_32FC1);
        for (var y = 0; y < featureMap.Height; y++)
        {
            for (var x = 0; x < featureMap.Width; x++)
            {
                var idx = y * featureMap.Width + x;
                scoreMap.Set(y, x, patchScores[idx]);
            }
        }

        using var scoreFull = new Mat();
        Cv2.Resize(
            scoreMap,
            scoreFull,
            new Size(_config.ImageSize, _config.ImageSize),
            0,
            0,
            InterpolationFlags.Cubic);

        var maxAbove = threshold;
        for (var y = 0; y < scoreFull.Rows; y++)
        {
            for (var x = 0; x < scoreFull.Cols; x++)
            {
                var value = scoreFull.At<float>(y, x);
                if (value > threshold && value > maxAbove)
                    maxAbove = value;
            }
        }

        using var original = Cv2.ImRead(imagePath, ImreadModes.Color);
        using var resized = new Mat();
        Cv2.Resize(original, resized, new Size(_config.ImageSize, _config.ImageSize));

        using var output = resized.Clone();

        if (maxAbove > threshold)
        {
            using var highlightNorm = new Mat(scoreFull.Size(), MatType.CV_32FC1, Scalar.All(0));
            var span = maxAbove - threshold;
            for (var y = 0; y < scoreFull.Rows; y++)
            {
                for (var x = 0; x < scoreFull.Cols; x++)
                {
                    var value = scoreFull.At<float>(y, x);
                    if (value > threshold)
                        highlightNorm.Set(y, x, (value - threshold) / span);
                }
            }

            using var mask = new Mat();
            Cv2.Compare(scoreFull, new Scalar(threshold), mask, CmpType.GT);

            using var highlightU8 = new Mat();
            highlightNorm.ConvertTo(highlightU8, MatType.CV_8UC1, 255.0);
            using var colored = new Mat();
            Cv2.ApplyColorMap(highlightU8, colored, ColormapTypes.Jet);

            using var blended = new Mat();
            Cv2.AddWeighted(resized, 0.55, colored, 0.45, 0, blended);
            blended.CopyTo(output, mask);
        }

        var fileName = $"{Path.GetFileNameWithoutExtension(imagePath)}_{label}_{score:F4}.jpg";
        var savePath = Path.Combine(outputDir, fileName);
        Cv2.ImWrite(savePath, output);
        return savePath;
    }

    public void Dispose() => _extractor.Dispose();

    private float GetThreshold() =>
        _config.UseManualThreshold ? _config.AnomalyThreshold : _model.AnomalyThreshold;
}
