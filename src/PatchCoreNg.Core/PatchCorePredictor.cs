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

        _extractor = new FeatureExtractor(_config.BackboneOnnxPath, _config.ImageSize);
        _memoryBank = new MemoryBank(model.MemoryBank);
    }

    public PredictionResult Predict(string imagePath, string? outputDir = null)
    {
        var tensor = ImagePreprocessor.LoadAndPreprocess(imagePath, _config.ImageSize);
        var featureMap = _extractor.Extract(tensor);
        var patches = LocalAggregator.Aggregate(featureMap, _config.PatchSize);
        var (_, imageScore, _) = _memoryBank.Score(patches, _config.NumNeighbors);

        var isAnomaly = imageScore > GetThreshold();
        var label = isAnomaly ? "NG" : "OK";

        string? heatmapPath = null;
        if (!string.IsNullOrWhiteSpace(outputDir))
        {
            Directory.CreateDirectory(outputDir);
            heatmapPath = SaveHeatmap(imagePath, featureMap, patches, outputDir, imageScore, label);
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
        float[][] patches,
        string outputDir,
        float score,
        string label)
    {
        var patchScores = _memoryBank.Score(patches, _config.NumNeighbors).Distances;

        var scoreMap = new Mat(featureMap.Height, featureMap.Width, MatType.CV_32FC1);
        for (var y = 0; y < featureMap.Height; y++)
        {
            for (var x = 0; x < featureMap.Width; x++)
            {
                var idx = y * featureMap.Width + x;
                scoreMap.Set(y, x, patchScores[idx]);
            }
        }

        Cv2.MinMaxLoc(scoreMap, out double _, out double maxVal);
        if (maxVal > 0)
            scoreMap /= maxVal;

        using var scoreU8 = new Mat();
        scoreMap.ConvertTo(scoreU8, MatType.CV_8UC1, 255.0);
        using var heatmap = new Mat();
        Cv2.ApplyColorMap(scoreU8, heatmap, ColormapTypes.Jet);

        using var original = Cv2.ImRead(imagePath, ImreadModes.Color);
        using var resized = new Mat();
        Cv2.Resize(original, resized, new Size(_config.ImageSize, _config.ImageSize));
        using var heatmapResized = new Mat();
        Cv2.Resize(heatmap, heatmapResized, new Size(_config.ImageSize, _config.ImageSize));
        using var overlay = new Mat();
        Cv2.AddWeighted(resized, 0.6, heatmapResized, 0.4, 0, overlay);

        var fileName = $"{Path.GetFileNameWithoutExtension(imagePath)}_{label}_{score:F4}.jpg";
        var savePath = Path.Combine(outputDir, fileName);
        Cv2.ImWrite(savePath, overlay);
        return savePath;
    }

    public void Dispose() => _extractor.Dispose();

    private float GetThreshold() =>
        _config.UseManualThreshold ? _config.AnomalyThreshold : _model.AnomalyThreshold;
}
