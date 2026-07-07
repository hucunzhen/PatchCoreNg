namespace PatchCoreNg;

public sealed record BackboneOption(
    string Id,
    string DisplayName,
    string OnnxRelativePath,
    string Description)
{
    public string GetResolvedOnnxPath() => AppPaths.Resolve(OnnxRelativePath);

    public bool IsOnnxAvailable() => File.Exists(ResolveAvailableOnnxPath());

    public string ResolveAvailableOnnxPath()
    {
        var path = GetResolvedOnnxPath();
        if (File.Exists(path))
            return path;

        if (string.Equals(Id, BackboneCatalog.DefaultId, StringComparison.OrdinalIgnoreCase))
        {
            var legacy = AppPaths.Resolve("models/wideresnet50_features.onnx");
            if (File.Exists(legacy))
                return legacy;
        }

        return path;
    }
}

public static class BackboneCatalog
{
    public const string DefaultId = "wide_resnet50_2";
    public const string CustomId = "custom";

    public static IReadOnlyList<BackboneOption> All { get; } =
    [
        new(
            DefaultId,
            "WideResNet-50 (推荐)",
            "models/wide_resnet50_2_features.onnx",
            "PatchCore 默认 backbone，精度与速度平衡"),
        new(
            "wide_resnet101_2",
            "WideResNet-101",
            "models/wide_resnet101_2_features.onnx",
            "更高精度，推理较慢"),
        new(
            "resnet50",
            "ResNet-50",
            "models/resnet50_features.onnx",
            "经典 backbone，工业场景常用"),
        new(
            "resnet101",
            "ResNet-101",
            "models/resnet101_features.onnx",
            "比 ResNet-50 更深，精度更高"),
        new(
            "resnet18",
            "ResNet-18",
            "models/resnet18_features.onnx",
            "轻量快速，适合边缘设备"),
        new(
            "efficientnet_b0",
            "EfficientNet-B0",
            "models/efficientnet_b0_features.onnx",
            "高效网络，速度较快"),
        new(
            "mobilenet_v3_large",
            "MobileNet-V3-Large",
            "models/mobilenet_v3_large_features.onnx",
            "移动端友好，轻量快速"),
        new(
            "mobilenet_v3_small",
            "MobileNet-V3-Small",
            "models/mobilenet_v3_small_features.onnx",
            "比 V3-Large 更小更快，适合极致提速"),
        new(
            CustomId,
            "自定义 ONNX",
            "",
            "手动指定任意 ONNX 特征提取模型路径")
    ];

    private static readonly Dictionary<string, BackboneOption> ById =
        All.ToDictionary(x => x.Id, StringComparer.OrdinalIgnoreCase);

    public static BackboneOption Get(string? id)
    {
        if (!string.IsNullOrWhiteSpace(id) && ById.TryGetValue(id, out var option))
            return option;

        return ById[DefaultId];
    }

    public static BackboneOption GetByOnnxPath(string onnxPath)
    {
        var resolved = AppPaths.Resolve(onnxPath);
        foreach (var option in All)
        {
            if (option.Id == CustomId || string.IsNullOrWhiteSpace(option.OnnxRelativePath))
                continue;

            if (string.Equals(option.ResolveAvailableOnnxPath(), resolved, StringComparison.OrdinalIgnoreCase))
                return option;
        }

        return ById[CustomId];
    }

    public static string ResolveOnnxPath(string backboneId, string? customOnnxPath = null)
    {
        var option = Get(backboneId);
        if (option.Id == CustomId)
        {
            if (string.IsNullOrWhiteSpace(customOnnxPath))
                throw new InvalidOperationException("自定义 backbone 需要指定 ONNX 路径。");

            return AppPaths.Resolve(customOnnxPath);
        }

        return option.ResolveAvailableOnnxPath();
    }

    public static string GetStatusText(string backboneId, string? customOnnxPath = null)
    {
        try
        {
            var path = ResolveOnnxPath(backboneId, customOnnxPath);
            return File.Exists(path) ? $"ONNX 已就绪: {Path.GetFileName(path)}" : "ONNX 未导出，请运行 export_backbone.py";
        }
        catch
        {
            return "请配置 ONNX 路径";
        }
    }

    public static string GetExportCommand(string? backboneId = null, int? targetDim = null, int? imageSize = null)
    {
        var dimSuffix = targetDim is > 0 ? $" --target-dim {targetDim.Value}" : string.Empty;
        var sizeSuffix = imageSize is > 0 ? $" --image-size {imageSize.Value}" : string.Empty;
        var extra = $"{dimSuffix}{sizeSuffix}";

        if (string.IsNullOrWhiteSpace(backboneId) || string.Equals(backboneId, CustomId, StringComparison.OrdinalIgnoreCase))
            return $"python scripts/export_backbone.py --all{extra}";

        return $"python scripts/export_backbone.py --backbone {backboneId}{extra}";
    }
}
