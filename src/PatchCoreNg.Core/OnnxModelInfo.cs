using Microsoft.ML.OnnxRuntime;

namespace PatchCoreNg;

public enum OnnxPrecision
{
    Float32 = 0,
    Float16 = 1,
    Int8 = 2,
}

public sealed class OnnxModelInfo
{
    public PreprocessMode PreprocessMode { get; init; }
    public OnnxPrecision Precision { get; init; }

    public static OnnxModelInfo FromSession(InferenceSession session, string onnxPath)
    {
        var meta = session.ModelMetadata.CustomMetadataMap;
        var preprocess = meta.TryGetValue("patchcore_preprocess", out var p) && p == "fused"
            ? PreprocessMode.FusedOnnx
            : PreprocessMode.Standard;

        var precision = OnnxPrecision.Float32;
        if (meta.TryGetValue("patchcore_precision", out var prec))
        {
            precision = prec switch
            {
                "fp16" => OnnxPrecision.Float16,
                "int8" => OnnxPrecision.Int8,
                _ => OnnxPrecision.Float32,
            };
        }
        else
        {
            precision = InferPrecisionFromPath(onnxPath);
            if (preprocess == PreprocessMode.Standard)
                preprocess = InferPreprocessFromPath(onnxPath);
        }

        return new OnnxModelInfo
        {
            PreprocessMode = preprocess,
            Precision = precision,
        };
    }

    public static OnnxModelInfo FromPath(string onnxPath)
    {
        using var session = new InferenceSession(onnxPath);
        return FromSession(session, onnxPath);
    }

    private static PreprocessMode InferPreprocessFromPath(string path)
    {
        var name = Path.GetFileName(path);
        return name.Contains("_fused", StringComparison.OrdinalIgnoreCase)
            ? PreprocessMode.FusedOnnx
            : PreprocessMode.Standard;
    }

    private static OnnxPrecision InferPrecisionFromPath(string path)
    {
        var name = Path.GetFileName(path);
        if (name.Contains("_int8", StringComparison.OrdinalIgnoreCase))
            return OnnxPrecision.Int8;
        if (name.Contains("_fp16", StringComparison.OrdinalIgnoreCase))
            return OnnxPrecision.Float16;
        return OnnxPrecision.Float32;
    }
}
