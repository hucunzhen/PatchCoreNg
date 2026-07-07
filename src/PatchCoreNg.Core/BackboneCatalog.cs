namespace PatchCoreNg;

public sealed record BackboneOption(
    string Id,
    string DisplayName,
    string OnnxRelativePath,
    string Description)
{
    public string GetResolvedOnnxPath() => AppPaths.Resolve(OnnxRelativePath);

    public bool IsOnnxAvailable() => File.Exists(ResolveAvailableOnnxPath());

    public string ResolveAvailableOnnxPath(bool preferGpu = false)
    {
        if (Id == BackboneCatalog.CustomId || string.IsNullOrWhiteSpace(OnnxRelativePath))
            return GetResolvedOnnxPath();

        var basePath = GetResolvedOnnxPath();
        var preferred = BackboneCatalog.ResolvePreferredOnnxVariant(basePath, preferGpu);
        if (File.Exists(preferred))
            return preferred;

        if (string.Equals(Id, BackboneCatalog.DefaultId, StringComparison.OrdinalIgnoreCase))
        {
            var legacy = AppPaths.Resolve("models/wideresnet50_features.onnx");
            if (File.Exists(legacy))
                return legacy;
        }

        return basePath;
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

    public static string ResolveOnnxPath(string backboneId, string? customOnnxPath = null, bool preferGpu = false)
    {
        var option = Get(backboneId);
        if (option.Id == CustomId)
        {
            if (string.IsNullOrWhiteSpace(customOnnxPath))
                throw new InvalidOperationException("自定义 backbone 需要指定 ONNX 路径。");

            return ResolvePreferredOnnxVariant(AppPaths.Resolve(customOnnxPath), preferGpu);
        }

        return option.ResolveAvailableOnnxPath(preferGpu);
    }

    public static string GetStatusText(string backboneId, string? customOnnxPath = null, bool preferGpu = false)
    {
        try
        {
            var path = ResolveOnnxPath(backboneId, customOnnxPath, preferGpu);
            if (!File.Exists(path))
                return "ONNX 未导出，请点击「导出 ONNX」";

            var fileName = Path.GetFileName(path);
            var hint = preferGpu && fileName.Contains("_int8", StringComparison.OrdinalIgnoreCase)
                ? "（GPU 下 INT8 可能更慢，建议导出 FP16）"
                : string.Empty;
            return $"ONNX 已就绪: {fileName}{hint}";
        }
        catch
        {
            return "请配置 ONNX 路径";
        }
    }

    public static string GetExportCommand(
        string? backboneId = null,
        int? targetDim = null,
        int? imageSize = null,
        bool legacyPreprocess = false,
        bool fp16 = false,
        bool int8 = false)
    {
        var dimSuffix = targetDim is > 0 ? $" --target-dim {targetDim.Value}" : string.Empty;
        var sizeSuffix = imageSize is > 0 ? $" --image-size {imageSize.Value}" : string.Empty;
        var legacySuffix = legacyPreprocess ? " --no-fuse-preprocess" : string.Empty;
        var fp16Suffix = fp16 ? " --fp16" : string.Empty;
        var int8Suffix = int8 ? " --int8" : string.Empty;
        var extra = $"{dimSuffix}{sizeSuffix}{legacySuffix}{fp16Suffix}{int8Suffix}";

        if (string.IsNullOrWhiteSpace(backboneId) || string.Equals(backboneId, CustomId, StringComparison.OrdinalIgnoreCase))
            return $"python scripts/export_backbone.py --all{extra}";

        return $"python scripts/export_backbone.py --backbone {backboneId}{extra}";
    }

    /// <summary>
    /// 选用 fused 变体。GPU(DirectML/CUDA) 优先 FP16（INT8 在 DirectML 上常更慢）；CPU 优先 INT8。
    /// </summary>
    public static string ResolvePreferredOnnxVariant(string baseOnnxPath, bool preferGpu = false)
    {
        var dir = Path.GetDirectoryName(baseOnnxPath) ?? string.Empty;
        var fileName = Path.GetFileNameWithoutExtension(baseOnnxPath);
        var stem = StripVariantSuffix(fileName);

        string[] candidates = preferGpu
            ?
            [
                $"{stem}_fused_fp16.onnx",
                $"{stem}_fused.onnx",
                $"{stem}.onnx",
                $"{stem}_fused_fp16_int8.onnx",
                $"{stem}_fused_int8.onnx",
            ]
            :
            [
                $"{stem}_fused_int8.onnx",
                $"{stem}_fused_fp16_int8.onnx",
                $"{stem}_fused_fp16.onnx",
                $"{stem}_fused.onnx",
                $"{stem}.onnx",
            ];

        foreach (var candidate in candidates)
        {
            var path = Path.Combine(dir, candidate);
            if (File.Exists(path))
                return path;
        }

        return baseOnnxPath;
    }

    /// <summary>
    /// 与 export_backbone.py 的 resolve_output_path 一致，用于导出完成后校验文件。
    /// </summary>
    public static string ResolveExportOutputPath(
        string backboneId,
        string modelsDirectory,
        bool fusePreprocess = true,
        bool fp16 = false,
        bool int8 = false)
    {
        var option = Get(backboneId);
        var stem = StripVariantSuffix(Path.GetFileNameWithoutExtension(option.OnnxRelativePath));
        var suffix = fusePreprocess ? "_fused" : string.Empty;
        if (fp16)
            suffix += "_fp16";
        if (int8)
            suffix += "_int8";
        return Path.Combine(modelsDirectory, $"{stem}{suffix}.onnx");
    }

    private static string StripVariantSuffix(string stem)
    {
        if (stem.EndsWith("_fused_fp16_int8", StringComparison.OrdinalIgnoreCase))
            return stem[..^"_fused_fp16_int8".Length];
        if (stem.EndsWith("_fused_int8", StringComparison.OrdinalIgnoreCase))
            return stem[..^"_fused_int8".Length];
        if (stem.EndsWith("_fused_fp16", StringComparison.OrdinalIgnoreCase))
            return stem[..^"_fused_fp16".Length];
        if (stem.EndsWith("_fused", StringComparison.OrdinalIgnoreCase))
            return stem[..^"_fused".Length];
        if (stem.EndsWith("_int8", StringComparison.OrdinalIgnoreCase))
            return stem[..^"_int8".Length];
        if (stem.EndsWith("_fp16", StringComparison.OrdinalIgnoreCase))
            return stem[..^"_fp16".Length];
        return stem;
    }
}
