using System.Numerics;
using System.Runtime.CompilerServices;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace PatchCoreNg;

internal static class ImageNetNormalizer
{
    private static readonly float[] InvStd =
    [
        1f / PatchCoreConfig.ImageNetStd[0],
        1f / PatchCoreConfig.ImageNetStd[1],
        1f / PatchCoreConfig.ImageNetStd[2],
    ];

    /// <summary>
    /// Fused ONNX 输入：OpenCV resize + CHW，RGB float 0~255（归一化在 ONNX 内）。
    /// </summary>
    public static float[] PreprocessFusedInput(Mat bgr, int imageSize)
    {
        using var blob = CvDnn.BlobFromImage(
            bgr,
            scaleFactor: 1.0,
            size: new Size(imageSize, imageSize),
            mean: Scalar.All(0),
            swapRB: true,
            crop: false);

        var planeSize = imageSize * imageSize;
        var tensor = new float[3 * planeSize];
        System.Runtime.InteropServices.Marshal.Copy(blob.Data, tensor, 0, tensor.Length);
        return tensor;
    }

    /// <summary>
    /// OpenCV DNN blob (scale + mean + resize + CHW) then SIMD /std.
    /// </summary>
    public static float[] PreprocessNormalized(Mat bgr, int imageSize)
    {
        using var blob = CvDnn.BlobFromImage(
            bgr,
            scaleFactor: 1.0 / 255.0,
            size: new Size(imageSize, imageSize),
            mean: new Scalar(
                PatchCoreConfig.ImageNetMean[0],
                PatchCoreConfig.ImageNetMean[1],
                PatchCoreConfig.ImageNetMean[2]),
            swapRB: true,
            crop: false);

        var planeSize = imageSize * imageSize;
        var tensor = new float[3 * planeSize];
        System.Runtime.InteropServices.Marshal.Copy(blob.Data, tensor, 0, tensor.Length);
        ApplyStdInPlace(tensor, planeSize);
        return tensor;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplyStdInPlace(Span<float> chw, int planeSize)
    {
        var simdWidth = Vector<float>.Count;
        for (var c = 0; c < 3; c++)
        {
            var inv = InvStd[c];
            var invVec = new Vector<float>(inv);
            var plane = chw.Slice(c * planeSize, planeSize);
            var i = 0;
            for (; i <= plane.Length - simdWidth; i += simdWidth)
            {
                var vec = new Vector<float>(plane.Slice(i, simdWidth));
                (vec * invVec).CopyTo(plane.Slice(i, simdWidth));
            }

            for (; i < plane.Length; i++)
                plane[i] *= inv;
        }
    }
}
