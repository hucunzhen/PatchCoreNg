using System.Diagnostics;
using System.Text;

namespace PatchCoreNg;

public sealed record BackboneExportOptions(
    bool Fp16 = false,
    bool Int8 = false,
    bool NoOnnxSim = false);

public sealed record BackboneExportResult(bool Success, string Message, string? OutputPath);

public static class BackboneExporter
{
    public static async Task<BackboneExportResult> ExportAsync(
        string backboneId,
        int imageSize = 224,
        int targetDim = 1024,
        BackboneExportOptions? options = null,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new BackboneExportOptions();
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
                "未找到 Python。请安装 Python 3，并执行:\n  pip install -r scripts/requirements-export.txt",
                null);
        }

        var modelsDir = Path.Combine(root, "models");
        Directory.CreateDirectory(modelsDir);
        var expectedPath = AppPaths.Resolve(
            BackboneCatalog.ResolveExportOutputPath(backboneId, modelsDir, fp16: options.Fp16, int8: options.Int8));

        var args = new StringBuilder()
            .Append('"').Append(script).Append('"')
            .Append(" --backbone ").Append(option.Id)
            .Append(" --models-dir \"").Append(modelsDir).Append('"')
            .Append(" --image-size ").Append(imageSize)
            .Append(" --target-dim ").Append(targetDim);

        if (options.Fp16)
            args.Append(" --fp16");
        if (options.Int8)
            args.Append(" --int8");
        if (options.NoOnnxSim)
            args.Append(" --no-onnxsim");

        var argsText = args.ToString();
        var flags = DescribeExportFlags(options);
        progress?.Report($"正在导出 {option.DisplayName} ONNX ({flags})...");
        progress?.Report($"命令: python {argsText}");

        using var process = new Process();
        process.StartInfo = new ProcessStartInfo
        {
            FileName = python,
            Arguments = argsText,
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
            return new BackboneExportResult(
                false,
                $"ONNX 导出失败:\n{detail}\n\n请确认已安装依赖:\n  pip install -r scripts/requirements-export.txt",
                null);
        }

        var outputPath = File.Exists(expectedPath)
            ? expectedPath
            : BackboneCatalog.ResolvePreferredOnnxVariant(
                AppPaths.Resolve(option.OnnxRelativePath),
                preferGpu: false);

        if (!File.Exists(outputPath))
        {
            return new BackboneExportResult(
                false,
                $"导出完成但未找到 ONNX 文件。\n期望: {expectedPath}",
                null);
        }

        var info = OnnxModelInfo.FromPath(outputPath);
        progress?.Report($"ONNX 已就绪: {outputPath}");
        progress?.Report($"预处理={info.PreprocessMode}, 精度={info.Precision}");
        return new BackboneExportResult(
            true,
            $"已导出: {Path.GetFileName(outputPath)}\n路径: {outputPath}\n预处理={info.PreprocessMode}, 精度={info.Precision}",
            outputPath);
    }

    private static string DescribeExportFlags(BackboneExportOptions options)
    {
        var parts = new List<string> { "fused预处理" };
        if (!options.NoOnnxSim)
            parts.Add("onnxsim");
        if (options.Int8)
            parts.Add("int8");
        if (options.Fp16)
            parts.Add("fp16");
        if (!options.Fp16 && !options.Int8)
            parts.Add("fp32");
        return string.Join(", ", parts);
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
