namespace PatchCoreNg;

public sealed class PatchCoreConfig
{
    public string BackboneId { get; init; } = BackboneCatalog.DefaultId;
    public string CustomBackboneOnnxPath { get; init; } = string.Empty;
    public string BackboneOnnxPath { get; init; } = "models/wide_resnet50_2_features.onnx";
    public int ImageSize { get; init; } = 224;
    public int PatchSize { get; init; } = 3;
    public int NumNeighbors { get; init; } = 9;
    public double CoresetRatio { get; init; } = 0.1;
    public int TargetEmbedDimension { get; init; } = 1024;
    public float AnomalyThreshold { get; init; } = 0.5f;
    public bool UseManualThreshold { get; init; }
    public bool UseGpu { get; init; } = true;
    public int GpuDeviceId { get; init; }
    public int InferenceBatchSize { get; init; } = 8;
    public int PreprocessParallelism { get; init; }
    public bool SaveHeatmap { get; init; } = true;
    public DistanceMetric DistanceMetric { get; init; } = DistanceMetric.SquaredEuclidean;
    public bool UseSimdDistance { get; init; } = true;
    public int PatchScoreParallelism { get; init; }
    public int FeatureMapDownscale { get; init; } = 1;
    public bool UseApproximateNearestNeighbors { get; init; }
    public int AnnClusterCount { get; init; } = 32;
    public int AnnProbeClusters { get; init; } = 4;

    public static readonly float[] ImageNetMean = [0.485f, 0.456f, 0.406f];
    public static readonly float[] ImageNetStd = [0.229f, 0.224f, 0.225f];
}
