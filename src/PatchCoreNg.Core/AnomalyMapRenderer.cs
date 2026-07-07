using OpenCvSharp;

namespace PatchCoreNg;

/// <summary>
/// 将特征图尺度的 patch 分数映射回原图坐标，全图 min-max 归一化后渲染异常热力图。
/// </summary>
public static class AnomalyMapRenderer
{
    private const double GaussianSigma = 4.0;
    private const double OverlayAlpha = 0.4;

    public static string SaveThresholdHeatmap(
        string imagePath,
        FeatureMap featureMap,
        float[] patchScores,
        string outputDir,
        float threshold,
        float imageScore,
        string label)
    {
        using var original = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (original.Empty())
            throw new FileNotFoundException($"无法读取图像: {imagePath}");

        using var scoreLow = BuildScoreMap(featureMap.Height, featureMap.Width, patchScores);
        using var output = RenderOverlay(original, scoreLow);

        var fileName = $"{Path.GetFileNameWithoutExtension(imagePath)}_{label}_{imageScore:F4}.jpg";
        var savePath = Path.Combine(outputDir, fileName);
        Cv2.ImWrite(savePath, output);
        return savePath;
    }

    public static Mat BuildScoreMap(int height, int width, float[] patchScores)
    {
        if (patchScores.Length != height * width)
        {
            throw new ArgumentException(
                $"patch 分数数量 {patchScores.Length} 与特征图尺寸 {width}x{height} 不匹配。");
        }

        var scoreMap = new Mat(height, width, MatType.CV_32FC1);
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var idx = y * width + x;
                scoreMap.Set(y, x, patchScores[idx]);
            }
        }

        return scoreMap;
    }

    public static Mat RenderOverlay(Mat originalBgr, Mat scoreLow)
    {
        var origW = originalBgr.Width;
        var origH = originalBgr.Height;

        using var scoreSmooth = new Mat();
        var kernel = ComputeGaussianKernelSize(scoreLow.Rows, scoreLow.Cols);
        Cv2.GaussianBlur(scoreLow, scoreSmooth, kernel, GaussianSigma);

        using var scoreFull = new Mat();
        Cv2.Resize(
            scoreSmooth,
            scoreFull,
            new Size(origW, origH),
            0,
            0,
            InterpolationFlags.Linear);

        Cv2.MinMaxLoc(scoreFull, out double minVal, out double maxVal);
        var span = (float)(maxVal - minVal);
        if (span <= 1e-8f)
            return originalBgr.Clone();

        using var normalized = new Mat();
        scoreFull.ConvertTo(normalized, MatType.CV_32FC1, 1.0 / span, -minVal / span);

        using var normalizedU8 = new Mat();
        normalized.ConvertTo(normalizedU8, MatType.CV_8UC1, 255.0);

        using var colored = new Mat();
        Cv2.ApplyColorMap(normalizedU8, colored, ColormapTypes.Jet);

        using var blended = new Mat();
        Cv2.AddWeighted(originalBgr, 1.0 - OverlayAlpha, colored, OverlayAlpha, 0, blended);
        return blended.Clone();
    }

    private static Size ComputeGaussianKernelSize(int height, int width)
    {
        var minSide = Math.Min(height, width);
        var k = (int)Math.Round(GaussianSigma * 6);
        if (k % 2 == 0)
            k++;
        k = Math.Clamp(k, 3, minSide | 1);
        return new Size(k, k);
    }
}
