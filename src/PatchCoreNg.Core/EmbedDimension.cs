namespace PatchCoreNg;

public static class EmbedDimension
{
    public const int Default = 1024;
    public const int Min = 32;
    public const int Max = 4096;

    public static void ValidateConfigValue(int targetEmbedDimension)
    {
        if (targetEmbedDimension < Min || targetEmbedDimension > Max)
        {
            throw new ArgumentOutOfRangeException(
                nameof(targetEmbedDimension),
                $"特征维度须在 {Min}~{Max} 之间，当前为 {targetEmbedDimension}。");
        }
    }

    public static void EnsureFeatureMapMatches(
        FeatureMap map,
        int expectedDimension,
        string onnxPath,
        string? backboneId = null,
        int imageSize = 224)
    {
        ValidateConfigValue(expectedDimension);

        if (map.Channels == expectedDimension)
            return;

        throw new InvalidOperationException(
            $"Backbone 输出通道数为 {map.Channels}，与配置 TargetEmbedDimension={expectedDimension} 不一致。\n" +
            $"ONNX: {onnxPath}\n" +
            $"请重新导出 backbone：\n  {BackboneCatalog.GetExportCommand(backboneId, expectedDimension, imageSize)}");
    }

    public static void EnsureMemoryBankMatches(PatchCoreModel model, string? modelPath = null)
    {
        if (model.MemoryBank.Length == 0)
            return;

        var bankDim = model.MemoryBank[0].Length;
        if (model.TargetEmbedDimension <= 0 || bankDim == model.TargetEmbedDimension)
            return;

        var pathHint = string.IsNullOrWhiteSpace(modelPath) ? string.Empty : $"\n模型: {modelPath}";
        throw new InvalidDataException(
            $"Memory Bank 向量维度为 {bankDim}，与模型 TargetEmbedDimension={model.TargetEmbedDimension} 不一致。{pathHint}\n" +
            "请使用匹配的 backbone 维度和模型重新训练。");
    }
}
