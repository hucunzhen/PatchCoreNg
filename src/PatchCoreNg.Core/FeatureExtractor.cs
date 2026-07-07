using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PatchCoreNg;

public sealed class FeatureExtractor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions = new();
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly int _imageSize;
    private readonly int _inputElementCount;
    private readonly bool _useIoBinding;
    private int _outputSpatial;
    private int _outputChannels;
    private int _outputHeight;
    private int _outputWidth;
    private bool _singleRunReady;

    private float[]? _singleInputBuffer;
    private long[]? _singleInputShape;
    private DenseTensor<float>? _singleInputTensor;
    private float[]? _singleOutputBuffer;
    private long[]? _singleOutputShape;
    private OrtValue? _singleInputOrt;
    private OrtValue? _singleOutputOrt;
    private OrtIoBinding? _singleIoBinding;

    public string ExecutionProvider { get; }
    public string ModelPath { get; }
    public string ModelFileName => Path.GetFileName(ModelPath);
    public PreprocessMode PreprocessMode { get; }
    public OnnxPrecision Precision { get; }
    public string SingleRunMode => _useIoBinding ? "IOBinding" : "GpuRun+BufferReuse";

    public string DescribeRuntime()
    {
        var text =
            $"模型={ModelFileName}, 设备={ExecutionProvider}, 单张={SingleRunMode}, " +
            $"预处理={PreprocessMode}, 精度={Precision}";
        if (IsGpuExecutionProvider(ExecutionProvider) && Precision == OnnxPrecision.Int8)
            text += " | 提示: GPU 下 INT8 常比 FP16 慢，建议改用 _fused_fp16";
        return text;
    }

    public FeatureExtractor(string onnxPath, int imageSize, bool useGpu = true, int gpuDeviceId = 0)
    {
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException(
                $"未找到 backbone ONNX 模型: {onnxPath}\n请先运行: python scripts/export_backbone.py --list");

        ModelPath = Path.GetFullPath(onnxPath);

        _imageSize = imageSize;
        _inputElementCount = 3 * imageSize * imageSize;

        var created = OnnxSessionFactory.Create(onnxPath, useGpu, gpuDeviceId);
        _session = created.Session;
        ExecutionProvider = created.Provider;
        _useIoBinding = !IsGpuExecutionProvider(ExecutionProvider);
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();

        var info = OnnxModelInfo.FromSession(_session, onnxPath);
        PreprocessMode = info.PreprocessMode;
        Precision = info.Precision;

        Warmup();
    }

    public FeatureMap Extract(ImageTensor input)
    {
        if (input.Data.Length != _inputElementCount)
        {
            throw new ArgumentException(
                $"ONNX 输入尺寸应为 {_inputElementCount}，实际 {input.Data.Length}。");
        }

        return RunSingle(input.Data);
    }

    public FeatureMap Extract(float[] imageTensor) =>
        Extract(ImageTensor.Standard(imageTensor, _imageSize));

    public IReadOnlyList<FeatureMap> ExtractBatch(IReadOnlyList<ImageTensor> inputs)
    {
        if (inputs.Count == 0)
            return [];

        if (inputs.Count == 1)
            return [Extract(inputs[0])];

        return ExtractMulti(inputs);
    }

    public IReadOnlyList<FeatureMap> ExtractBatch(IReadOnlyList<float[]> imageTensors)
    {
        var wrapped = new ImageTensor[imageTensors.Count];
        for (var i = 0; i < imageTensors.Count; i++)
            wrapped[i] = ImageTensor.Standard(imageTensors[i], _imageSize);

        return ExtractBatch(wrapped);
    }

    private List<FeatureMap> ExtractMulti(IReadOnlyList<ImageTensor> inputs)
    {
        var batch = inputs.Count;
        var combined = new float[batch * _inputElementCount];
        for (var i = 0; i < batch; i++)
        {
            if (inputs[i].Data.Length != _inputElementCount)
            {
                throw new ArgumentException(
                    $"ONNX 输入尺寸应为 {_inputElementCount}，实际 {inputs[i].Data.Length}。");
            }

            inputs[i].Data.AsSpan().CopyTo(combined.AsSpan(i * _inputElementCount, _inputElementCount));
        }

        return RunMulti(combined, batch);
    }

    private FeatureMap RunSingle(float[] input)
    {
        EnsureSingleRunReady();
        input.AsSpan().CopyTo(_singleInputBuffer!);
        return ExecuteSingleRun();
    }

    private List<FeatureMap> RunMulti(float[] combined, int batch)
    {
        EnsureSingleRunReady();

        var inputTensor = new DenseTensor<float>(combined, [batch, 3, _imageSize, _imageSize]);
        var inputValue = NamedOnnxValue.CreateFromTensor(_inputName, inputTensor);
        using var results = _session.Run([inputValue]);

        var tensor = results.First().AsTensor<float>();
        var flat = tensor.ToArray();
        var maps = new List<FeatureMap>(batch);
        for (var i = 0; i < batch; i++)
        {
            var slice = new float[_outputSpatial];
            Array.Copy(flat, i * _outputSpatial, slice, 0, _outputSpatial);
            maps.Add(new FeatureMap(slice, _outputChannels, _outputHeight, _outputWidth));
        }

        return maps;
    }

    private void Warmup()
    {
        if (!TryCacheOutputShapeFromMetadata())
        {
            var probeInput = new float[_inputElementCount];
            var inputTensor = new DenseTensor<float>(probeInput, [1, 3, _imageSize, _imageSize]);
            var inputValue = NamedOnnxValue.CreateFromTensor(_inputName, inputTensor);
            using var results = _session.Run([inputValue]);
            CacheOutputShape(results.First().AsTensor<float>());
        }

        InitSingleRunInfrastructure();
        Array.Clear(_singleInputBuffer!);
        _ = ExecuteSingleRun();
    }

    private void EnsureSingleRunReady()
    {
        if (_singleRunReady)
            return;

        Warmup();
    }

    private void InitSingleRunInfrastructure()
    {
        _singleInputBuffer = new float[_inputElementCount];
        _singleInputShape = [1, 3, _imageSize, _imageSize];

        if (_useIoBinding)
        {
            _singleInputOrt = OrtValue.CreateTensorValueFromMemory(_singleInputBuffer, _singleInputShape);
            _singleOutputBuffer = new float[_outputSpatial];
            _singleOutputShape = [1, _outputChannels, _outputHeight, _outputWidth];
            _singleOutputOrt = OrtValue.CreateTensorValueFromMemory(_singleOutputBuffer, _singleOutputShape);

            _singleIoBinding = _session.CreateIoBinding();
            _singleIoBinding.BindInput(_inputName, _singleInputOrt);
            _singleIoBinding.BindOutput(_outputName, _singleOutputOrt);
        }
        else
        {
            _singleInputTensor = new DenseTensor<float>(_singleInputBuffer, [1, 3, _imageSize, _imageSize]);
        }

        _singleRunReady = true;
    }

    private FeatureMap ExecuteSingleRun()
    {
        if (_useIoBinding)
        {
            _session.RunWithBinding(_runOptions, _singleIoBinding!);
            var result = new float[_outputSpatial];
            Array.Copy(_singleOutputBuffer!, result, _outputSpatial);
            return new FeatureMap(result, _outputChannels, _outputHeight, _outputWidth);
        }

        var inputValue = NamedOnnxValue.CreateFromTensor(_inputName, _singleInputTensor!);
        using var results = _session.Run([inputValue]);
        return new FeatureMap(
            results.First().AsTensor<float>().ToArray(),
            _outputChannels,
            _outputHeight,
            _outputWidth);
    }

    private static bool IsGpuExecutionProvider(string provider) =>
        provider.StartsWith("DirectML:", StringComparison.OrdinalIgnoreCase)
        || provider.StartsWith("CUDA:", StringComparison.OrdinalIgnoreCase);

    private bool TryCacheOutputShapeFromMetadata()
    {
        var dims = _session.OutputMetadata[_outputName].Dimensions;
        if (dims.Length != 4)
            return false;

        if (dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0)
            return false;

        _outputChannels = (int)dims[1];
        _outputHeight = (int)dims[2];
        _outputWidth = (int)dims[3];
        _outputSpatial = _outputChannels * _outputHeight * _outputWidth;
        return true;
    }

    private void CacheOutputShape(Tensor<float> output)
    {
        var dims = output.Dimensions.ToArray();
        if (dims.Length != 4)
            throw new InvalidOperationException($"ONNX 输出维度异常: [{string.Join(", ", dims)}]");

        _outputChannels = dims[1];
        _outputHeight = dims[2];
        _outputWidth = dims[3];
        _outputSpatial = _outputChannels * _outputHeight * _outputWidth;
    }

    public void Dispose()
    {
        _singleIoBinding?.Dispose();
        _singleInputOrt?.Dispose();
        _singleOutputOrt?.Dispose();
        _runOptions.Dispose();
        _session.Dispose();
    }
}

public readonly record struct FeatureMap(float[] Data, int Channels, int Height, int Width)
{
    public float Get(int c, int y, int x) => Data[c * Height * Width + y * Width + x];
}
