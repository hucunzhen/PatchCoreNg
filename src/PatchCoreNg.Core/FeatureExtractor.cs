using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PatchCoreNg;

public sealed class FeatureExtractor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _imageSize;

    public FeatureExtractor(string onnxPath, int imageSize)
    {
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException(
                $"未找到 backbone ONNX 模型: {onnxPath}\n请先运行: python scripts/export_backbone.py --list");

        _session = new InferenceSession(onnxPath);
        _inputName = _session.InputMetadata.Keys.First();
        _imageSize = imageSize;
    }

    public FeatureMap Extract(float[] imageTensor)
    {
        var inputShape = new[] { 1, 3, _imageSize, _imageSize };
        var input = NamedOnnxValue.CreateFromTensor(
            _inputName,
            new DenseTensor<float>(imageTensor, inputShape));

        using var results = _session.Run([input]);
        var output = results.First().AsTensor<float>().ToArray();
        var dims = results.First().AsTensor<float>().Dimensions.ToArray();

        // Expected output: [1, C, H, W]
        var channels = dims[^3];
        var height = dims[^2];
        var width = dims[^1];

        return new FeatureMap(output, channels, height, width);
    }

    public void Dispose() => _session.Dispose();
}

public readonly record struct FeatureMap(float[] Data, int Channels, int Height, int Width)
{
    public float Get(int c, int y, int x) => Data[c * Height * Width + y * Width + x];
}
