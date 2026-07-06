using System.Text.Json;

namespace PatchCoreNg;

public sealed class PatchCoreModel
{
    public int Version { get; init; } = 1;
    public string? BackboneId { get; init; }
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
            BackboneId = config.BackboneId,
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
        if (string.Equals(Path.GetFileName(path), ProfileOutputLayout.ConfigFileName, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"所选文件是配置 config.json，不是模型 patchcore_model.json: {path}\n" +
                "请在模型目录中选择 patchcore_model.json，或重新训练。");
        }

        var json = File.ReadAllText(path);
        var model = JsonSerializer.Deserialize<PatchCoreModel>(json)
            ?? throw new InvalidDataException($"无法解析模型: {path}");

        if (model.MemoryBank is null || model.MemoryBank.Length == 0)
        {
            throw new InvalidDataException(
                $"模型 Memory Bank 为空: {path}\n" +
                "请确认选择的是 patchcore_model.json（不是 config.json），并重新完成训练。");
        }

        if (model.MemoryBank[0] is null || model.MemoryBank[0].Length == 0)
            throw new InvalidDataException($"模型 Memory Bank 向量维度为 0: {path}");

        return model;
    }

    public PatchCoreModel WithTunedParams(float threshold, int numNeighbors) => new()
    {
        Version = Version,
        BackboneId = BackboneId,
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
