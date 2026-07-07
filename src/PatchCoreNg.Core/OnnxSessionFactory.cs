using Microsoft.ML.OnnxRuntime;

namespace PatchCoreNg;

public static class OnnxSessionFactory
{
    public static (InferenceSession Session, string Provider) Create(
        string onnxPath,
        bool preferGpu,
        int gpuDeviceId = 0)
    {
        if (preferGpu)
        {
            try
            {
                return (CreateSession(onnxPath, ConfigureGpuOptions(gpuDeviceId, useCuda: true)), $"CUDA:{gpuDeviceId}");
            }
            catch (Exception ex)
            {
                try
                {
                    return (CreateSession(onnxPath, ConfigureGpuOptions(gpuDeviceId, useCuda: false)), $"DirectML:{gpuDeviceId}");
                }
                catch
                {
                    var session = CreateSession(onnxPath, ConfigureCpuOptions());
                    return (session, $"CPU (GPU 不可用: {ex.Message})");
                }
            }
        }

        return (CreateSession(onnxPath, ConfigureCpuOptions()), "CPU");
    }

    private static InferenceSession CreateSession(string onnxPath, SessionOptions options)
    {
        try
        {
            return new InferenceSession(onnxPath, options);
        }
        finally
        {
            options.Dispose();
        }
    }

    private static SessionOptions ConfigureCommonOptions()
    {
        var threads = Math.Clamp(Environment.ProcessorCount / 2, 1, 8);
        return new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            EnableMemoryPattern = true,
            EnableCpuMemArena = true,
            IntraOpNumThreads = threads,
            InterOpNumThreads = 1,
        };
    }

    private static SessionOptions ConfigureGpuOptions(int gpuDeviceId, bool useCuda)
    {
        var options = ConfigureCommonOptions();
        if (useCuda)
            options.AppendExecutionProvider_CUDA(gpuDeviceId);
        else
            options.AppendExecutionProvider_DML(gpuDeviceId);
        return options;
    }

    private static SessionOptions ConfigureCpuOptions() => ConfigureCommonOptions();
}
