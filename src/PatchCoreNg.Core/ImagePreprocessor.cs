using OpenCvSharp;

namespace PatchCoreNg;

public enum PreprocessMode
{
    /// <summary>Resize + ImageNet normalize in C# (standard ONNX input).</summary>
    Standard = 0,

    /// <summary>RGB 0~255 CHW; resize + normalize inside fused ONNX.</summary>
    FusedOnnx = 1,
}

public readonly record struct ImageTensor(float[] Data, int Height, int Width, PreprocessMode Mode)
{
    public int Channels => 3;

    public bool IsFused => Mode == PreprocessMode.FusedOnnx;

    public static ImageTensor Standard(float[] normalizedChw, int imageSize) =>
        new(normalizedChw, imageSize, imageSize, PreprocessMode.Standard);
}

public static class ImagePreprocessor
{
    public static ImageTensor LoadAndPreprocess(string imagePath, int imageSize, PreprocessMode mode = PreprocessMode.Standard)
    {
        using var bgr = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (bgr.Empty())
            throw new FileNotFoundException($"无法读取图像: {imagePath}");

        return Preprocess(bgr, imageSize, mode);
    }

    public static ImageTensor Preprocess(Mat bgr, int imageSize, PreprocessMode mode = PreprocessMode.Standard)
    {
        if (mode == PreprocessMode.FusedOnnx)
        {
            var data = ImageNetNormalizer.PreprocessFusedInput(bgr, imageSize);
            return new ImageTensor(data, imageSize, imageSize, PreprocessMode.FusedOnnx);
        }

        var tensor = ImageNetNormalizer.PreprocessNormalized(bgr, imageSize);
        return ImageTensor.Standard(tensor, imageSize);
    }

    public static float[] LoadAndPreprocessLegacy(string imagePath, int imageSize) =>
        LoadAndPreprocess(imagePath, imageSize, PreprocessMode.Standard).Data;

    public static IEnumerable<string> EnumerateImages(string path)
    {
        if (File.Exists(path))
            return [path];

        if (!Directory.Exists(path))
            throw new DirectoryNotFoundException($"路径不存在: {path}");

        var extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
        };

        return Directory.EnumerateFiles(path, "*.*", SearchOption.AllDirectories)
            .Where(f => extensions.Contains(Path.GetExtension(f)))
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase);
    }
}
