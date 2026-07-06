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
    public string OkTrainPath { get; set; } = string.Empty;
    public string OkTunePath { get; set; } = string.Empty;
    public string NgTunePath { get; set; } = string.Empty;
    public string ModelOutputPath { get; set; } = "models/patchcore_model.json";
    public string InferModelPath { get; set; } = "models/patchcore_model.json";
    public bool AutoSearchNeighbors { get; set; } = true;

    public static PatchCoreSettings CreateDefault()
    {
        var settings = new PatchCoreSettings();
        settings.SyncBackbonePath();
        return settings;
    }

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
        BackboneOnnxPath = BackboneCatalog.ResolveOnnxPath(BackboneId, CustomBackboneOnnxPath),
        ImageSize = ImageSize,
        PatchSize = PatchSize,
        NumNeighbors = NumNeighbors,
        CoresetRatio = CoresetRatio,
        TargetEmbedDimension = TargetEmbedDimension,
        AnomalyThreshold = AnomalyThreshold,
        UseManualThreshold = UseManualThreshold
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
        var json = JsonSerializer.Serialize(settings, JsonOptions);
        File.WriteAllText(path, json);
    }

    public static PatchCoreSettings Load(string filePath)
    {
        var json = File.ReadAllText(filePath);
        var settings = JsonSerializer.Deserialize<PatchCoreSettings>(json)
            ?? throw new InvalidDataException($"无法解析参数配置: {filePath}");
        settings.ProfileName = Path.GetFileNameWithoutExtension(filePath);
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
