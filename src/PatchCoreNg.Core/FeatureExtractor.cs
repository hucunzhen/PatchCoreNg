using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PatchCoreNg;

public sealed class FeatureExtractor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _imageSize;

    public string ExecutionProvider { get; }

    public FeatureExtractor(string onnxPath, int imageSize, bool useGpu = true, int gpuDeviceId = 0)
    {
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException(
                $"未找到 backbone ONNX 模型: {onnxPath}\n请先运行: python scripts/export_backbone.py --list");

        var created = OnnxSessionFactory.Create(onnxPath, useGpu, gpuDeviceId);
        _session = created.Session;
        ExecutionProvider = created.Provider;
        _inputName = _session.InputMetadata.Keys.First();
        _imageSize = imageSize;
    }

    public FeatureMap Extract(float[] imageTensor) => ExtractBatch([imageTensor])[0];

    public IReadOnlyList<FeatureMap> ExtractBatch(IReadOnlyList<float[]> imageTensors)
    {
        if (imageTensors.Count == 0)
            return [];

        var pixelsPerImage = 3 * _imageSize * _imageSize;
        var batch = imageTensors.Count;
        var combined = new float[batch * pixelsPerImage];
        for (var i = 0; i < batch; i++)
        {
            if (imageTensors[i].Length != pixelsPerImage)
                throw new ArgumentException($"图像张量尺寸不匹配: 期望 {pixelsPerImage}, 实际 {imageTensors[i].Length}");

            imageTensors[i].AsSpan().CopyTo(combined.AsSpan(i * pixelsPerImage, pixelsPerImage));
        }

        var inputShape = new[] { batch, 3, _imageSize, _imageSize };
        var input = NamedOnnxValue.CreateFromTensor(
            _inputName,
            new DenseTensor<float>(combined, inputShape));

        using var results = _session.Run([input]);
        var tensor = results.First().AsTensor<float>();
        var dims = tensor.Dimensions.ToArray();

        if (dims.Length != 4)
            throw new InvalidOperationException($"ONNX 输出维度异常: [{string.Join(", ", dims)}]");

        var channels = dims[1];
        var height = dims[2];
        var width = dims[3];
        var spatial = channels * height * width;
        var output = tensor.ToArray();
        var maps = new List<FeatureMap>(batch);

        for (var i = 0; i < batch; i++)
        {
            var slice = new float[spatial];
            output.AsSpan(i * spatial, spatial).CopyTo(slice);
            maps.Add(new FeatureMap(slice, channels, height, width));
        }

        return maps;
    }

    public void Dispose() => _session.Dispose();
}

public readonly record struct FeatureMap(float[] Data, int Channels, int Height, int Width)
{
    public float Get(int c, int y, int x) => Data[c * Height * Width + y * Width + x];
}
