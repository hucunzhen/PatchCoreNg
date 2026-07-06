namespace PatchCoreNg;

public static class FeatureMapDownscaler
{
    public static FeatureMap Downscale(FeatureMap map, int factor)
    {
        if (factor <= 1)
            return map;

        if (map.Height % factor != 0 || map.Width % factor != 0)
        {
            throw new InvalidOperationException(
                $"特征图尺寸 {map.Width}x{map.Height} 无法被降采样倍数 {factor} 整除。");
        }

        var outH = map.Height / factor;
        var outW = map.Width / factor;
        var data = new float[map.Channels * outH * outW];

        for (var oy = 0; oy < outH; oy++)
        {
            for (var ox = 0; ox < outW; ox++)
            {
                for (var c = 0; c < map.Channels; c++)
                {
                    var sum = 0f;
                    for (var dy = 0; dy < factor; dy++)
                    {
                        for (var dx = 0; dx < factor; dx++)
                        {
                            sum += map.Get(c, oy * factor + dy, ox * factor + dx);
                        }
                    }

                    var idx = c * outH * outW + oy * outW + ox;
                    data[idx] = sum / (factor * factor);
                }
            }
        }

        return new FeatureMap(data, map.Channels, outH, outW);
    }
}
