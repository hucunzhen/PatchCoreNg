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
    private readonly KnnsSearchOptions _knnsOptions;

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
            AnomalyThreshold = model.AnomalyThreshold,
        };

        _knnsOptions = KnnsSearchOptions.FromModel(model, _config);
        _extractor = new FeatureExtractor(
            _config.BackboneOnnxPath,
            _config.ImageSize,
            _config.UseGpu,
            _config.GpuDeviceId);
        _memoryBank = new MemoryBank(model.MemoryBank, _knnsOptions);
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
        var detail = KnnsFeatureHelper.ScoreFeatureMap(
            _memoryBank,
            featureMap,
            _config.PatchSize,
            _config.NumNeighbors,
            _knnsOptions);
        return CreateResult(imagePath, detail, outputDir);
    }

    public PredictionResult CreateResult(
        string imagePath,
        FeatureMapScoreDetail scoreDetail,
        string? outputDir = null) =>
        CreateResult(imagePath, scoreDetail.ScoredFeatureMap, scoreDetail.ImageScore, scoreDetail.PatchDistances, outputDir);

    public PredictionResult CreateResult(
        string imagePath,
        FeatureMap scoredFeatureMap,
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
            heatmapPath = AnomalyMapRenderer.SaveThresholdHeatmap(
                imagePath,
                scoredFeatureMap,
                patchDistances,
                outputDir,
                GetThreshold(),
                imageScore,
                label);
        }

        return new PredictionResult
        {
            ImagePath = imagePath,
            AnomalyScore = imageScore,
            IsAnomaly = isAnomaly,
            Label = label,
            HeatmapPath = heatmapPath,
        };
    }

    public void Dispose() => _extractor.Dispose();

    private float GetThreshold() =>
        _config.UseManualThreshold ? _config.AnomalyThreshold : _model.AnomalyThreshold;
}
