using Microsoft.ML.OnnxRuntime;

namespace PatchCoreNg;

public static class OnnxSessionFactory
{
    public static (InferenceSession Session, string Provider) Create(
        string onnxPath,
        bool preferGpu,
        int gpuDeviceId = 0)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };

        if (preferGpu)
        {
            try
            {
                options.AppendExecutionProvider_CUDA(gpuDeviceId);
                return (new InferenceSession(onnxPath, options), $"CUDA:{gpuDeviceId}");
            }
            catch (Exception ex)
            {
                options.Dispose();
                options = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                };

                try
                {
                    options.AppendExecutionProvider_DML(gpuDeviceId);
                    return (new InferenceSession(onnxPath, options), $"DirectML:{gpuDeviceId}");
                }
                catch
                {
                    options.Dispose();
                    options = new SessionOptions
                    {
                        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    };
                    var session = new InferenceSession(onnxPath, options);
                    return (session, $"CPU (GPU 不可用: {ex.Message})");
                }
            }
        }

        return (new InferenceSession(onnxPath, options), "CPU");
    }
}
