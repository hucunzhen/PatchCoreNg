using System.Diagnostics;
using OpenCvSharp;

namespace PatchCoreNg;

public sealed class PredictionResult
{
    public required string ImagePath { get; init; }
    public required float AnomalyScore { get; init; }
    public required bool IsAnomaly { get; init; }
    public required string Label { get; init; }
    public string? HeatmapPath { get; init; }
    public PredictionStageTiming? StageTiming { get; init; }
}

public sealed class PatchCorePredictor : IDisposable
{
    private readonly PatchCoreConfig _config;
    private readonly FeatureExtractor _extractor;
    private readonly MemoryBank _memoryBank;
    private readonly PatchCoreModel _model;
    private readonly KnnsSearchOptions _knnsOptions;

    public string ExecutionProvider => _extractor.ExecutionProvider;

    public string KnnsBackend => _memoryBank.UsesNativeAcceleration
        ? "Native(C++)"
        : NativeAcceleration.IsAvailable
            ? "Managed(C#)"
            : "Managed(C#), native 未加载";

    public string OnnxRuntimeInfo =>
        $"{_extractor.ModelFileName} | {_extractor.PreprocessMode}/{_extractor.Precision} | {_extractor.SingleRunMode}";

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
        var timing = new PredictionStageTimingBuilder();
        var watch = Stopwatch.StartNew();

        var tensor = ImagePreprocessor.LoadAndPreprocess(imagePath, _config.ImageSize, _extractor.PreprocessMode);
        timing.Preprocess = watch.Elapsed;
        watch.Restart();

        var featureMap = _extractor.Extract(tensor);
        timing.FeatureExtract = watch.Elapsed;
        EmbedDimension.EnsureFeatureMapMatches(
            featureMap,
            _config.TargetEmbedDimension,
            _config.BackboneOnnxPath,
            _config.BackboneId,
            _config.ImageSize);

        return ScoreAndCreateResult(imagePath, featureMap, outputDir, timing);
    }

    public PredictionResult PredictFromFeatureMap(
        string imagePath,
        FeatureMap featureMap,
        string? outputDir = null) =>
        ScoreAndCreateResult(imagePath, featureMap, outputDir, new PredictionStageTimingBuilder());

    public PredictionResult CreateResult(
        string imagePath,
        FeatureMapScoreDetail scoreDetail,
        string? outputDir = null) =>
        FinalizeResult(
            imagePath,
            scoreDetail.ScoredFeatureMap,
            scoreDetail.ImageScore,
            scoreDetail.PatchDistances,
            outputDir,
            new PredictionStageTimingBuilder());

    public PredictionResult CreateResult(
        string imagePath,
        FeatureMap scoredFeatureMap,
        float imageScore,
        float[] patchDistances,
        string? outputDir = null) =>
        FinalizeResult(imagePath, scoredFeatureMap, imageScore, patchDistances, outputDir, new PredictionStageTimingBuilder());

    public void Dispose()
    {
        _memoryBank.Dispose();
        _extractor.Dispose();
    }

    private PredictionResult ScoreAndCreateResult(
        string imagePath,
        FeatureMap featureMap,
        string? outputDir,
        PredictionStageTimingBuilder timing)
    {
        var watch = Stopwatch.StartNew();

        var prepared = KnnsFeatureHelper.PrepareFeatureMap(featureMap, _knnsOptions);
        timing.FeatureDownscale = watch.Elapsed;
        watch.Restart();

        if (prepared.Height == 0 || prepared.Width == 0)
        {
            throw new InvalidOperationException(
                $"特征图为空 ({prepared.Channels}x{prepared.Height}x{prepared.Width})。");
        }

        float[] patchDistances;
        float imageScore;
        if (_memoryBank.UsesNativeAcceleration)
        {
            (patchDistances, imageScore) = _memoryBank.ScoreFeatureMap(
                prepared,
                _config.PatchSize,
                _config.NumNeighbors);
            timing.PatchAggregate = TimeSpan.Zero;
            timing.KnnsScore = watch.Elapsed;
        }
        else
        {
            var patches = KnnsFeatureHelper.AggregatePatches(prepared, _config.PatchSize);
            timing.PatchAggregate = watch.Elapsed;
            watch.Restart();

            (patchDistances, imageScore) = KnnsFeatureHelper.ScorePatches(
                _memoryBank,
                patches,
                _config.NumNeighbors);
            timing.KnnsScore = watch.Elapsed;
        }

        return FinalizeResult(imagePath, prepared, imageScore, patchDistances, outputDir, timing);
    }

    private PredictionResult FinalizeResult(
        string imagePath,
        FeatureMap scoredFeatureMap,
        float imageScore,
        float[] patchDistances,
        string? outputDir,
        PredictionStageTimingBuilder timing)
    {
        var watch = Stopwatch.StartNew();

        if (imageScore <= 0f && _memoryBank.Embeddings.Length == 0)
        {
            throw new InvalidOperationException(
                "Memory Bank 为空，推理分数恒为 0。请使用 patchcore_model.json 并重新训练。");
        }

        var isAnomaly = imageScore > GetThreshold();
        var label = isAnomaly ? "NG" : "OK";
        timing.Judgment = watch.Elapsed;
        watch.Restart();

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

        timing.Heatmap = watch.Elapsed;

        return new PredictionResult
        {
            ImagePath = imagePath,
            AnomalyScore = imageScore,
            IsAnomaly = isAnomaly,
            Label = label,
            HeatmapPath = heatmapPath,
            StageTiming = timing.Build(),
        };
    }

    private float GetThreshold() =>
        _config.UseManualThreshold ? _config.AnomalyThreshold : _model.AnomalyThreshold;
}

internal sealed class PredictionStageTimingBuilder
{
    public TimeSpan Preprocess { get; set; }
    public TimeSpan FeatureExtract { get; set; }
    public TimeSpan FeatureDownscale { get; set; }
    public TimeSpan PatchAggregate { get; set; }
    public TimeSpan KnnsScore { get; set; }
    public TimeSpan Judgment { get; set; }
    public TimeSpan Heatmap { get; set; }

    public PredictionStageTiming Build() => new()
    {
        Preprocess = Preprocess,
        FeatureExtract = FeatureExtract,
        FeatureDownscale = FeatureDownscale,
        PatchAggregate = PatchAggregate,
        KnnsScore = KnnsScore,
        Judgment = Judgment,
        Heatmap = Heatmap,
    };
}
