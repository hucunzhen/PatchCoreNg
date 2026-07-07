using System.Text.Json;

namespace PatchCoreNg;

public sealed class PatchCoreSettings
{
    public string ProfileName { get; set; } = "default";
    public string BackboneId { get; set; } = BackboneCatalog.DefaultId;
    public string CustomBackboneOnnxPath { get; set; } = string.Empty;
    public string BackboneOnnxPath { get; set; } = "models/wide_resnet50_2_features.onnx";
    public int ImageSize { get; set; } = 224;
    public int PatchSize { get; set; } = 3;
    public int NumNeighbors { get; set; } = 9;
    public double CoresetRatio { get; set; } = 0.1;
    public int TargetEmbedDimension { get; set; } = 1024;
    public float AnomalyThreshold { get; set; } = 0.5f;
    public bool UseManualThreshold { get; set; }
    public string OkDataPath { get; set; } = string.Empty;
    public string NgDataPath { get; set; } = string.Empty;
    public double OkTrainRatio { get; set; } = 0.6;
    public double OkTuneRatio { get; set; } = 0.2;
    public double OkTestRatio { get; set; } = 0.2;
    public double NgTuneRatio { get; set; } = 0.5;
    public DatasetSplitMode SplitMode { get; set; } = DatasetSplitMode.Ratio;
    public int OkMemoryCount { get; set; } = 6;
    public int OkTuneCount { get; set; } = 2;
    public int NgTuneCount { get; set; } = 3;
    public int SplitSeed { get; set; } = 42;
    public string ModelOutputDir { get; set; } = "models";
    public bool AutoSearchNeighbors { get; set; } = true;
    public bool AutoSearchAnnProbes { get; set; } = true;
    public bool UseGpu { get; set; } = true;
    public int GpuDeviceId { get; set; }
    public int InferenceBatchSize { get; set; } = 8;
    public int PreprocessParallelism { get; set; }
    public bool SaveHeatmap { get; set; } = true;
    public DistanceMetric DistanceMetric { get; set; } = DistanceMetric.SquaredEuclidean;
    public bool UseSimdDistance { get; set; } = true;
    public int PatchScoreParallelism { get; set; }
    public int FeatureMapDownscale { get; set; } = 1;
    public bool UseApproximateNearestNeighbors { get; set; }
    public int AnnClusterCount { get; set; } = 32;
    public int AnnProbeClusters { get; set; } = 4;

    // 兼容旧版：原为模型文件路径
    public string ModelOutputPath { get; set; } = string.Empty;
    public string InferModelPath { get; set; } = string.Empty;

    public string GetModelPathForProfile(string profileName) =>
        ProfileOutputLayout.GetModelPath(ModelOutputDir, profileName);

    public string GetConfigPathForProfile(string profileName) =>
        ProfileOutputLayout.GetConfigPath(ModelOutputDir, profileName);

    // 兼容旧版配置文件字段
    public string OkTrainPath { get; set; } = string.Empty;
    public string OkTunePath { get; set; } = string.Empty;
    public string NgTunePath { get; set; } = string.Empty;

    public static PatchCoreSettings CreateDefault()
    {
        var settings = new PatchCoreSettings();
        settings.SyncBackbonePath();
        return settings;
    }

    public void NormalizeLegacyPaths()
    {
        if (string.IsNullOrWhiteSpace(OkDataPath))
        {
            if (!string.IsNullOrWhiteSpace(OkTrainPath))
                OkDataPath = OkTrainPath;
            else if (!string.IsNullOrWhiteSpace(OkTunePath))
                OkDataPath = OkTunePath;
        }

        if (string.IsNullOrWhiteSpace(NgDataPath) && !string.IsNullOrWhiteSpace(NgTunePath))
            NgDataPath = NgTunePath;

        NormalizeSplitRatios();
        NormalizeModelOutputDir();
    }

    public void NormalizeModelOutputDir()
    {
        if (!string.IsNullOrWhiteSpace(ModelOutputDir))
        {
            ModelOutputDir = ProfileOutputLayout.NormalizeOutputBaseDir(ModelOutputDir);
            return;
        }

        if (!string.IsNullOrWhiteSpace(ModelOutputPath))
            ModelOutputDir = ProfileOutputLayout.NormalizeOutputBaseDir(ModelOutputPath);
        else if (!string.IsNullOrWhiteSpace(InferModelPath))
            ModelOutputDir = ProfileOutputLayout.NormalizeOutputBaseDir(InferModelPath);
        else
            ModelOutputDir = "models";
    }

    public void NormalizeSplitRatios()
    {
        var okSum = OkTrainRatio + OkTuneRatio + OkTestRatio;
        if (okSum <= 0)
        {
            OkTrainRatio = 0.6;
            OkTuneRatio = 0.2;
            OkTestRatio = 0.2;
        }
        else if (Math.Abs(okSum - 1.0) > 0.001)
        {
            OkTrainRatio /= okSum;
            OkTuneRatio /= okSum;
            OkTestRatio /= okSum;
        }

        NgTuneRatio = Math.Clamp(NgTuneRatio, 0.05, 0.95);
    }

    public DatasetSplitOptions ToSplitOptions() => new()
    {
        Mode = SplitMode,
        OkTrainRatio = OkTrainRatio,
        OkTuneRatio = OkTuneRatio,
        OkTestRatio = OkTestRatio,
        NgTuneRatio = NgTuneRatio,
        OkMemoryCount = OkMemoryCount,
        OkTuneCount = OkTuneCount,
        NgTuneCount = NgTuneCount,
        SplitSeed = SplitSeed
    };

    public void SyncBackbonePath()
    {
        if (string.Equals(BackboneId, BackboneCatalog.CustomId, StringComparison.OrdinalIgnoreCase))
        {
            if (!string.IsNullOrWhiteSpace(CustomBackboneOnnxPath))
                BackboneOnnxPath = CustomBackboneOnnxPath;
            return;
        }

        BackboneOnnxPath = BackboneCatalog.Get(BackboneId).OnnxRelativePath;
    }

    public PatchCoreConfig ToConfig() => new()
    {
        BackboneId = BackboneId,
        CustomBackboneOnnxPath = CustomBackboneOnnxPath,
        BackboneOnnxPath = BackboneCatalog.ResolveOnnxPath(BackboneId, CustomBackboneOnnxPath, UseGpu),
        ImageSize = ImageSize,
        PatchSize = PatchSize,
        NumNeighbors = NumNeighbors,
        CoresetRatio = CoresetRatio,
        TargetEmbedDimension = TargetEmbedDimension,
        AnomalyThreshold = AnomalyThreshold,
        UseManualThreshold = UseManualThreshold,
        UseGpu = UseGpu,
        GpuDeviceId = GpuDeviceId,
        InferenceBatchSize = InferenceBatchSize,
        PreprocessParallelism = PreprocessParallelism,
        SaveHeatmap = SaveHeatmap,
        DistanceMetric = DistanceMetric,
        UseSimdDistance = UseSimdDistance,
        PatchScoreParallelism = PatchScoreParallelism,
        FeatureMapDownscale = FeatureMapDownscale,
        UseApproximateNearestNeighbors = UseApproximateNearestNeighbors,
        AnnClusterCount = AnnClusterCount,
        AnnProbeClusters = AnnProbeClusters,
    };

    public void ApplyFrom(PatchCoreConfig config)
    {
        BackboneId = config.BackboneId;
        CustomBackboneOnnxPath = config.CustomBackboneOnnxPath;
        BackboneOnnxPath = config.BackboneOnnxPath;
        ImageSize = config.ImageSize;
        PatchSize = config.PatchSize;
        NumNeighbors = config.NumNeighbors;
        CoresetRatio = config.CoresetRatio;
        TargetEmbedDimension = config.TargetEmbedDimension;
        AnomalyThreshold = config.AnomalyThreshold;
        UseManualThreshold = config.UseManualThreshold;
        UseGpu = config.UseGpu;
        GpuDeviceId = config.GpuDeviceId;
        InferenceBatchSize = config.InferenceBatchSize;
        PreprocessParallelism = config.PreprocessParallelism;
        SaveHeatmap = config.SaveHeatmap;
        DistanceMetric = config.DistanceMetric;
        UseSimdDistance = config.UseSimdDistance;
        PatchScoreParallelism = config.PatchScoreParallelism;
        FeatureMapDownscale = config.FeatureMapDownscale;
        UseApproximateNearestNeighbors = config.UseApproximateNearestNeighbors;
        AnnClusterCount = config.AnnClusterCount;
        AnnProbeClusters = config.AnnProbeClusters;
    }
}

public static class SettingsStore
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public static string DefaultConfigDirectory => AppPaths.Resolve("config");

    public static string GetProfilePath(string configDirectory, string profileName)
    {
        var safeName = SanitizeProfileName(profileName);
        return Path.Combine(configDirectory, $"{safeName}.json");
    }

    public static void Save(string configDirectory, PatchCoreSettings settings)
    {
        Directory.CreateDirectory(configDirectory);
        var path = GetProfilePath(configDirectory, settings.ProfileName);
        SaveToFile(path, settings);
    }

    public static void SaveToFile(string filePath, PatchCoreSettings settings)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(filePath) ?? ".");
        var json = JsonSerializer.Serialize(settings, JsonOptions);
        File.WriteAllText(filePath, json);
    }

    public static PatchCoreSettings Load(string filePath)
    {
        var json = File.ReadAllText(filePath);
        var settings = JsonSerializer.Deserialize<PatchCoreSettings>(json)
            ?? throw new InvalidDataException($"无法解析参数配置: {filePath}");
        settings.ProfileName = Path.GetFileNameWithoutExtension(filePath);
        settings.NormalizeLegacyPaths();
        return settings;
    }

    public static IReadOnlyList<string> ListProfiles(string configDirectory)
    {
        if (!Directory.Exists(configDirectory))
            return [];

        return Directory.EnumerateFiles(configDirectory, "*.json", SearchOption.TopDirectoryOnly)
            .Select(Path.GetFileNameWithoutExtension)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Cast<string>()
            .OrderBy(name => name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public static void DeleteProfile(string configDirectory, string profileName)
    {
        var path = GetProfilePath(configDirectory, profileName);
        if (File.Exists(path))
            File.Delete(path);
    }

    public static void SaveLastActiveProfile(string configDirectory, string profileName)
    {
        Directory.CreateDirectory(configDirectory);
        var safeName = SanitizeProfileName(profileName);
        File.WriteAllText(Path.Combine(configDirectory, ".last_profile"), safeName);
    }

    public static string? LoadLastActiveProfile(string configDirectory)
    {
        var path = Path.Combine(configDirectory, ".last_profile");
        if (!File.Exists(path))
            return null;

        var name = File.ReadAllText(path).Trim();
        return string.IsNullOrWhiteSpace(name) ? null : name;
    }

    private static string SanitizeProfileName(string profileName)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = profileName.Trim().Where(ch => !invalid.Contains(ch)).ToArray();
        var sanitized = new string(chars);
        return string.IsNullOrWhiteSpace(sanitized) ? "default" : sanitized;
    }
}
