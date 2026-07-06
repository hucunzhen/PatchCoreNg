namespace PatchCoreNg;

public enum DatasetSplitMode
{
    Ratio,
    Count
}

public sealed record DatasetSplitOptions
{
    public DatasetSplitMode Mode { get; init; } = DatasetSplitMode.Ratio;
    public double OkTrainRatio { get; init; } = 0.6;
    public double OkTuneRatio { get; init; } = 0.2;
    public double OkTestRatio { get; init; } = 0.2;
    public double NgTuneRatio { get; init; } = 0.5;
    public int OkMemoryCount { get; init; }
    public int OkTuneCount { get; init; }
    public int NgTuneCount { get; init; }
    public int SplitSeed { get; init; } = 42;
}

public sealed record DatasetSplitResult(
    IReadOnlyList<string> OkTrainPaths,
    IReadOnlyList<string> OkTunePaths,
    IReadOnlyList<string> OkTestPaths,
    IReadOnlyList<string> NgTunePaths,
    IReadOnlyList<string> NgTestPaths)
{
    public bool CanTune => OkTunePaths.Count > 0 && NgTunePaths.Count > 0;

    public bool CanTest => OkTestPaths.Count > 0 && NgTestPaths.Count > 0;
}

public static class DatasetSplitter
{
    public static DatasetSplitResult Split(
        string okDataPath,
        string? ngDataPath,
        DatasetSplitOptions options)
    {
        var okImages = ImagePreprocessor.EnumerateImages(okDataPath).ToList();
        if (okImages.Count == 0)
            throw new InvalidOperationException($"OK 目录中没有图像: {okDataPath}");

        var okBuckets = options.Mode == DatasetSplitMode.Count
            ? SplitOkByCounts(okImages, options.OkMemoryCount, options.OkTuneCount, options.SplitSeed)
            : SplitOkByRatios(okImages, options.OkTrainRatio, options.OkTuneRatio, options.OkTestRatio, options.SplitSeed);

        var ngImages = string.IsNullOrWhiteSpace(ngDataPath)
            ? []
            : ImagePreprocessor.EnumerateImages(ngDataPath).ToList();

        var ngBuckets = ngImages.Count == 0
            ? new List<IReadOnlyList<string>> { Array.Empty<string>(), Array.Empty<string>() }
            : options.Mode == DatasetSplitMode.Count
                ? SplitNgByCounts(ngImages, options.NgTuneCount, options.SplitSeed + 1)
                : SplitNgByRatios(ngImages, options.NgTuneRatio, options.SplitSeed + 1);

        return new DatasetSplitResult(
            okBuckets[0],
            okBuckets[1],
            okBuckets[2],
            ngBuckets[0],
            ngBuckets[1]);
    }

    private static List<IReadOnlyList<string>> SplitOkByRatios(
        IReadOnlyList<string> images,
        double okTrainRatio,
        double okTuneRatio,
        double okTestRatio,
        int seed)
    {
        NormalizeRatios(ref okTrainRatio, ref okTuneRatio, ref okTestRatio);
        return SplitByRatios(
            images,
            [okTrainRatio, okTuneRatio, okTestRatio],
            seed,
            ensureFirstBucketMinCount: 1);
    }

    private static List<IReadOnlyList<string>> SplitOkByCounts(
        IReadOnlyList<string> images,
        int memoryCount,
        int tuneCount,
        int seed)
    {
        if (memoryCount < 1)
            throw new InvalidOperationException("固定数量模式下 OK Memory 数量至少为 1。");

        if (tuneCount < 0)
            throw new InvalidOperationException("OK 调参数量不能为负数。");

        var shuffled = Shuffle(images, seed);
        var n = shuffled.Count;
        var mem = Math.Min(memoryCount, n);
        if (n >= 1 && mem < 1)
            mem = 1;

        var tune = Math.Min(tuneCount, n - mem);
        var test = n - mem - tune;

        return
        [
            shuffled.Take(mem).ToList(),
            shuffled.Skip(mem).Take(tune).ToList(),
            shuffled.Skip(mem + tune).Take(test).ToList()
        ];
    }

    private static List<IReadOnlyList<string>> SplitNgByRatios(
        IReadOnlyList<string> images,
        double ngTuneRatio,
        int seed)
    {
        var ngTuneRatioClamped = Math.Clamp(ngTuneRatio, 0.05, 0.95);
        var ngTestRatio = 1.0 - ngTuneRatioClamped;
        return SplitByRatios(images, [ngTuneRatioClamped, ngTestRatio], seed, ensureFirstBucketMinCount: 0);
    }

    private static List<IReadOnlyList<string>> SplitNgByCounts(
        IReadOnlyList<string> images,
        int tuneCount,
        int seed)
    {
        if (tuneCount < 0)
            throw new InvalidOperationException("NG 调参数量不能为负数。");

        var shuffled = Shuffle(images, seed);
        var n = shuffled.Count;
        var tune = Math.Min(tuneCount, n);
        var test = n - tune;

        return
        [
            shuffled.Take(tune).ToList(),
            shuffled.Skip(tune).Take(test).ToList()
        ];
    }

    private static void NormalizeRatios(ref double a, ref double b, ref double c)
    {
        a = Math.Max(0, a);
        b = Math.Max(0, b);
        c = Math.Max(0, c);
        var sum = a + b + c;
        if (sum <= 0)
        {
            a = 0.6;
            b = 0.2;
            c = 0.2;
            return;
        }

        a /= sum;
        b /= sum;
        c /= sum;
    }

    private static List<IReadOnlyList<string>> SplitByRatios(
        IReadOnlyList<string> items,
        IReadOnlyList<double> ratios,
        int seed,
        int ensureFirstBucketMinCount)
    {
        if (ratios.Count == 0)
            throw new ArgumentException("至少需要一个划分比例。");

        if (items.Count == 0)
            return ratios.Select(_ => (IReadOnlyList<string>)Array.Empty<string>()).ToList();

        var shuffled = Shuffle(items, seed);
        var n = shuffled.Count;
        var sum = ratios.Sum();
        var normalized = ratios.Select(r => r / sum).ToArray();
        var counts = new int[normalized.Length];
        var used = 0;

        for (var i = 0; i < normalized.Length - 1; i++)
        {
            counts[i] = (int)Math.Floor(n * normalized[i]);
            used += counts[i];
        }

        counts[^1] = n - used;

        if (ensureFirstBucketMinCount > 0 && counts[0] < ensureFirstBucketMinCount && n >= ensureFirstBucketMinCount)
        {
            var need = ensureFirstBucketMinCount - counts[0];
            for (var i = counts.Length - 1; i >= 1 && need > 0; i--)
            {
                var take = Math.Min(need, counts[i]);
                counts[i] -= take;
                counts[0] += take;
                need -= take;
            }
        }

        var buckets = new List<IReadOnlyList<string>>();
        var offset = 0;
        for (var i = 0; i < counts.Length; i++)
        {
            buckets.Add(shuffled.Skip(offset).Take(counts[i]).ToList());
            offset += counts[i];
        }

        return buckets;
    }

    private static List<string> Shuffle(IReadOnlyList<string> items, int seed)
    {
        var list = items.ToList();
        var rng = new Random(seed);

        for (var i = list.Count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }

        return list;
    }
}
