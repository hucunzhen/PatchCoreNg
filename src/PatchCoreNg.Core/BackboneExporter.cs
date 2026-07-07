using System.Diagnostics;
using System.Text;

namespace PatchCoreNg;

public sealed record BackboneExportResult(bool Success, string Message, string? OutputPath);

public static class BackboneExporter
{
    public static async Task<BackboneExportResult> ExportAsync(
        string backboneId,
        int imageSize = 224,
        int targetDim = 1024,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var option = BackboneCatalog.Get(backboneId);
        if (option.Id == BackboneCatalog.CustomId)
            return new BackboneExportResult(false, "自定义 backbone 请手动指定 ONNX 路径。", null);

        var root = AppPaths.ProjectRoot;
        var script = Path.Combine(root, "scripts", "export_backbone.py");
        if (!File.Exists(script))
            return new BackboneExportResult(false, $"未找到导出脚本: {script}", null);

        var python = FindPythonExecutable();
        if (python is null)
        {
            return new BackboneExportResult(
                false,
                "未找到 Python。请安装 Python 3，并执行: pip install torch torchvision onnx",
                null);
        }

        var modelsDir = Path.Combine(root, "models");
        Directory.CreateDirectory(modelsDir);
        var outputPath = AppPaths.Resolve(option.OnnxRelativePath);

        var args = new StringBuilder()
            .Append('"').Append(script).Append('"')
            .Append(" --backbone ").Append(option.Id)
            .Append(" --models-dir \"").Append(modelsDir).Append('"')
            .Append(" --image-size ").Append(imageSize)
            .Append(" --target-dim ").Append(targetDim)
            .ToString();

        progress?.Report($"正在导出 {option.DisplayName} ONNX（首次会从 PyTorch 下载预训练权重）...");

        using var process = new Process();
        process.StartInfo = new ProcessStartInfo
        {
            FileName = python,
            Arguments = args,
            WorkingDirectory = root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        var stderr = new StringBuilder();
        process.OutputDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                progress?.Report(e.Data);
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                stderr.AppendLine(e.Data);
        };

        if (!process.Start())
            return new BackboneExportResult(false, "无法启动 Python 进程。", null);

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);

        if (process.ExitCode != 0)
        {
            var detail = stderr.Length > 0 ? stderr.ToString().Trim() : $"退出码 {process.ExitCode}";
            return new BackboneExportResult(false, $"ONNX 导出失败:\n{detail}", null);
        }

        if (!File.Exists(outputPath))
            return new BackboneExportResult(false, $"导出完成但未找到文件: {outputPath}", null);

        progress?.Report($"ONNX 已就绪: {outputPath}");
        return new BackboneExportResult(true, $"已导出: {outputPath}", outputPath);
    }

    private static string? FindPythonExecutable()
    {
        foreach (var (fileName, prefixArgs) in new[] { ("python", ""), ("py", "-3") })
        {
            try
            {
                using var process = Process.Start(new ProcessStartInfo
                {
                    FileName = fileName,
                    Arguments = $"{prefixArgs} -c \"import sys; print(sys.executable)\"".Trim(),
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                });

                if (process is null)
                    continue;

                var output = process.StandardOutput.ReadToEnd().Trim();
                process.WaitForExit(5000);
                if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
                    return output;
            }
            catch
            {
                // try next candidate
            }
        }

        return null;
    }
}
