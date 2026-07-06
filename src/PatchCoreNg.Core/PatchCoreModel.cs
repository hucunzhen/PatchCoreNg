using System.Text.Json;

namespace PatchCoreNg;

public sealed class PatchCoreModel
{
    public int Version { get; init; } = 1;
    public int ImageSize { get; init; }
    public int PatchSize { get; init; }
    public int NumNeighbors { get; init; }
    public double CoresetRatio { get; init; }
    public int TargetEmbedDimension { get; init; }
    public int FeatureHeight { get; init; }
    public int FeatureWidth { get; init; }
    public float AnomalyThreshold { get; init; }
    public float[][] MemoryBank { get; init; } = [];

    public static PatchCoreModel Create(
        PatchCoreConfig config,
        FeatureMap referenceMap,
        float[][] memoryBank,
        float anomalyThreshold)
    {
        return new PatchCoreModel
        {
            ImageSize = config.ImageSize,
            PatchSize = config.PatchSize,
            NumNeighbors = config.NumNeighbors,
            CoresetRatio = config.CoresetRatio,
            TargetEmbedDimension = config.TargetEmbedDimension,
            FeatureHeight = referenceMap.Height,
            FeatureWidth = referenceMap.Width,
            AnomalyThreshold = anomalyThreshold,
            MemoryBank = memoryBank
        };
    }

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        var json = JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    public static PatchCoreModel Load(string path)
    {
        var json = File.ReadAllText(path);
        var model = JsonSerializer.Deserialize<PatchCoreModel>(json)
            ?? throw new InvalidDataException($"无法解析模型: {path}");
        return model;
    }

    public PatchCoreModel WithTunedParams(float threshold, int numNeighbors) => new()
    {
        Version = Version,
        ImageSize = ImageSize,
        PatchSize = PatchSize,
        NumNeighbors = numNeighbors,
        CoresetRatio = CoresetRatio,
        TargetEmbedDimension = TargetEmbedDimension,
        FeatureHeight = FeatureHeight,
        FeatureWidth = FeatureWidth,
        AnomalyThreshold = threshold,
        MemoryBank = MemoryBank
    };
}
