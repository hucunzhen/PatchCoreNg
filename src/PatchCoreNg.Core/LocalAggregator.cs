namespace PatchCoreNg;

public static class LocalAggregator
{
    /// <summary>
    /// 对特征图做局部邻域平均聚合 (PatchCore patchsize x patchsize)。
    /// </summary>
    public static float[][] Aggregate(FeatureMap map, int patchSize)
    {
        var half = patchSize / 2;
        var patchCount = map.Height * map.Width;
        var patches = new float[patchCount][];

        for (var y = 0; y < map.Height; y++)
        {
            for (var x = 0; x < map.Width; x++)
            {
                var idx = y * map.Width + x;
                var aggregated = new float[map.Channels];
                var count = 0;

                for (var dy = -half; dy <= half; dy++)
                {
                    for (var dx = -half; dx <= half; dx++)
                    {
                        var ny = y + dy;
                        var nx = x + dx;
                        if (ny < 0 || ny >= map.Height || nx < 0 || nx >= map.Width)
                            continue;

                        for (var c = 0; c < map.Channels; c++)
                            aggregated[c] += map.Get(c, ny, nx);
                        count++;
                    }
                }

                for (var c = 0; c < map.Channels; c++)
                    aggregated[c] /= count;

                patches[idx] = aggregated;
            }
        }

        return patches;
    }
}
