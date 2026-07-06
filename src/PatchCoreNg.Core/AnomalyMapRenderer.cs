using OpenCvSharp;

namespace PatchCoreNg;

/// <summary>
/// 将特征图尺度的 patch 分数映射回原图坐标并渲染热力图叠加。
/// </summary>
public static class AnomalyMapRenderer
{
    private const double GaussianSigma = 4.0;

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
        using var output = RenderOverlay(original, scoreLow, threshold);

        var fileName = $"{Path.GetFileNameWithoutExtension(imagePath)}_{label}_{imageScore:F4}.jpg";
        var savePath = Path.Combine(outputDir, fileName);
        Cv2.ImWrite(savePath, output);
        return savePath;
    }

    public static Mat BuildScoreMap(int height, int width, float[] patchScores)
    {
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

    public static Mat RenderOverlay(Mat originalBgr, Mat scoreLow, float threshold)
    {
        var origW = originalBgr.Width;
        var origH = originalBgr.Height;

        using var scoreSmooth = new Mat();
        var kernel = ComputeGaussianKernelSize(scoreLow.Rows, scoreLow.Cols);
        Cv2.GaussianBlur(scoreLow, scoreSmooth, kernel, GaussianSigma);

        var maxAbove = threshold;
        for (var y = 0; y < scoreSmooth.Rows; y++)
        {
            for (var x = 0; x < scoreSmooth.Cols; x++)
            {
                var value = scoreSmooth.At<float>(y, x);
                if (value > threshold && value > maxAbove)
                    maxAbove = value;
            }
        }

        var output = originalBgr.Clone();
        if (maxAbove <= threshold)
            return output;

        using var highlightLow = new Mat(scoreLow.Size(), MatType.CV_32FC1, Scalar.All(0));
        var span = maxAbove - threshold;
        for (var y = 0; y < scoreSmooth.Rows; y++)
        {
            for (var x = 0; x < scoreSmooth.Cols; x++)
            {
                var value = scoreSmooth.At<float>(y, x);
                if (value > threshold)
                    highlightLow.Set(y, x, (value - threshold) / span);
            }
        }

        using var maskLow = new Mat();
        Cv2.Compare(scoreSmooth, new Scalar(threshold), maskLow, CmpType.GT);

        using var highlightFull = new Mat();
        Cv2.Resize(
            highlightLow,
            highlightFull,
            new Size(origW, origH),
            0,
            0,
            InterpolationFlags.Linear);

        using var maskFull = new Mat();
        Cv2.Resize(
            maskLow,
            maskFull,
            new Size(origW, origH),
            0,
            0,
            InterpolationFlags.Nearest);

        using var highlightU8 = new Mat();
        highlightFull.ConvertTo(highlightU8, MatType.CV_8UC1, 255.0);
        using var colored = new Mat();
        Cv2.ApplyColorMap(highlightU8, colored, ColormapTypes.Jet);

        using var blended = new Mat();
        Cv2.AddWeighted(originalBgr, 0.55, colored, 0.45, 0, blended);
        blended.CopyTo(output, maskFull);
        return output;
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
