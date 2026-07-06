using System.Diagnostics;

namespace PatchCoreNg;

public sealed class StepProgress
{
    private readonly IProgress<string>? _progress;
    private readonly Stopwatch _total = Stopwatch.StartNew();

    public StepProgress(IProgress<string>? progress)
    {
        _progress = progress;
    }

    public static string FormatElapsed(TimeSpan elapsed)
    {
        if (elapsed.TotalHours >= 1)
            return $"{elapsed.TotalHours:F1}h";
        if (elapsed.TotalMinutes >= 1)
            return $"{elapsed.TotalMinutes:F1}m";
        if (elapsed.TotalSeconds >= 1)
            return $"{elapsed.TotalSeconds:F1}s";
        return $"{elapsed.TotalMilliseconds:F0}ms";
    }

    public void Begin(string step, string? detail = null)
    {
        Report(FormatBegin(step, detail));
    }

    public void End(string step, TimeSpan elapsed, string? detail = null)
    {
        Report(FormatEnd(step, elapsed, detail));
    }

    public void Info(string step, string message)
    {
        Report($"[{step}] {message} | 累计 {FormatElapsed(_total.Elapsed)}");
    }

    public void Report(string message) => _progress?.Report(message);

    public T Run<T>(string step, Func<T> action, string? beginDetail = null, Func<T, string?>? endDetail = null)
    {
        Begin(step, beginDetail);
        var watch = Stopwatch.StartNew();
        try
        {
            var result = action();
            watch.Stop();
            End(step, watch.Elapsed, endDetail?.Invoke(result));
            return result;
        }
        catch (Exception ex)
        {
            watch.Stop();
            End(step, watch.Elapsed, $"失败: {ex.Message}");
            throw;
        }
    }

    public void Run(string step, Action action, string? beginDetail = null, string? endDetail = null)
    {
        Run<object?>(
            step,
            () =>
            {
                action();
                return null;
            },
            beginDetail,
            _ => endDetail);
    }

    public void Complete(string summary)
    {
        Report($"[完成] {summary} | 总耗时 {FormatElapsed(_total.Elapsed)}");
    }

    public TimeSpan TotalElapsed => _total.Elapsed;

    public static string FormatBegin(string step, string? detail = null) =>
        string.IsNullOrWhiteSpace(detail)
            ? $"[{step}] 开始..."
            : $"[{step}] 开始 | {detail}";

    public static string FormatEnd(string step, TimeSpan elapsed, string? detail = null) =>
        string.IsNullOrWhiteSpace(detail)
            ? $"[{step}] 完成 | 耗时 {FormatElapsed(elapsed)}"
            : $"[{step}] 完成 | 耗时 {FormatElapsed(elapsed)} | {detail}";
}
