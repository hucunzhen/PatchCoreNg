using OpenCvSharp;

namespace PatchCoreNg;

public static class ImagePreprocessor
{
    public static float[] LoadAndPreprocess(string imagePath, int imageSize)
    {
        using var bgr = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (bgr.Empty())
            throw new FileNotFoundException($"无法读取图像: {imagePath}");

        return Preprocess(bgr, imageSize);
    }

    public static float[] Preprocess(Mat bgr, int imageSize)
    {
        using var rgb = new Mat();
        Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);
        using var resized = new Mat();
        Cv2.Resize(rgb, resized, new Size(imageSize, imageSize));

        var tensor = new float[3 * imageSize * imageSize];
        for (var y = 0; y < imageSize; y++)
        {
            for (var x = 0; x < imageSize; x++)
            {
                var pixel = resized.At<Vec3b>(y, x);
                var r = (pixel.Item0 / 255f - PatchCoreConfig.ImageNetMean[0]) / PatchCoreConfig.ImageNetStd[0];
                var g = (pixel.Item1 / 255f - PatchCoreConfig.ImageNetMean[1]) / PatchCoreConfig.ImageNetStd[1];
                var b = (pixel.Item2 / 255f - PatchCoreConfig.ImageNetMean[2]) / PatchCoreConfig.ImageNetStd[2];

                tensor[0 * imageSize * imageSize + y * imageSize + x] = r;
                tensor[1 * imageSize * imageSize + y * imageSize + x] = g;
                tensor[2 * imageSize * imageSize + y * imageSize + x] = b;
            }
        }

        return tensor;
    }

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
