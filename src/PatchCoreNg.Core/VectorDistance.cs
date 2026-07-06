using System.Numerics;

namespace PatchCoreNg;

internal static class VectorDistance
{
    public static float Compute(ReadOnlySpan<float> a, ReadOnlySpan<float> b, KnnsSearchOptions options)
    {
        var squared = options.UseSimdDistance
            ? SquaredDistanceSimd(a, b)
            : SquaredDistanceScalar(a, b);

        return options.DistanceMetric == DistanceMetric.Euclidean
            ? MathF.Sqrt(squared)
            : squared;
    }

    public static float SquaredDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b, bool useSimd) =>
        useSimd ? SquaredDistanceSimd(a, b) : SquaredDistanceScalar(a, b);

    private static float SquaredDistanceScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"向量维度不匹配: {a.Length} vs {b.Length}");

        var sum = 0f;
        for (var i = 0; i < a.Length; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }

        return sum;
    }

    private static float SquaredDistanceSimd(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"向量维度不匹配: {a.Length} vs {b.Length}");

        var simdWidth = Vector<float>.Count;
        var sumVec = Vector<float>.Zero;
        var i = 0;

        for (; i <= a.Length - simdWidth; i += simdWidth)
        {
            var va = new Vector<float>(a.Slice(i, simdWidth));
            var vb = new Vector<float>(b.Slice(i, simdWidth));
            var diff = va - vb;
            sumVec += diff * diff;
        }

        var sum = 0f;
        for (var lane = 0; lane < simdWidth; lane++)
            sum += sumVec[lane];

        for (; i < a.Length; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }

        return sum;
    }
}
