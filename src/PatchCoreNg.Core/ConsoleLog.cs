using System.Runtime.InteropServices;
using System.Text;

namespace PatchCoreNg;

/// <summary>
/// 为 WinExe 附加独立控制台窗口，或将日志写入已有控制台（CLI）。
/// </summary>
public static class ConsoleLog
{
    private static readonly object Gate = new();
    private static bool _initialized;

    public static bool IsEnabled => _initialized;

    /// <summary>
    /// 启动日志控制台。WPF 会新建窗口；CLI 复用当前控制台。
    /// </summary>
    public static void Attach(string title = "PatchCore-NG Log")
    {
        lock (Gate)
        {
            if (_initialized)
                return;

            if (OperatingSystem.IsWindows())
            {
                if (GetConsoleWindow() == IntPtr.Zero)
                {
                    if (!AllocConsole())
                        return;
                }

                SetConsoleTitle(title);
            }

            ConfigureStreams();
            _initialized = true;
            WriteLineCore("日志控制台已就绪。");
        }
    }

    public static void WriteLine(string message)
    {
        if (string.IsNullOrEmpty(message))
            return;

        lock (Gate)
        {
            if (!_initialized)
                return;

            WriteLineCore(message);
        }
    }

    /// <summary>
    /// 同时写入控制台，并转发到 UI 进度回调（需在 UI 线程上创建以正确 marshal）。
    /// </summary>
    public static IProgress<string> CreateProgress(Action<string>? uiHandler = null)
    {
        IProgress<string>? uiProgress = uiHandler is null ? null : new Progress<string>(uiHandler);
        return new ForwardingProgress(uiProgress);
    }

    private static void WriteLineCore(string message)
    {
        try
        {
            Console.WriteLine($"[{DateTime.Now:HH:mm:ss.fff}] {message}");
        }
        catch
        {
            // 控制台已关闭时忽略
        }
    }

    private static void ConfigureStreams()
    {
        try
        {
            Console.OutputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
            Console.InputEncoding = Console.OutputEncoding;

            var stdout = new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true };
            var stderr = new StreamWriter(Console.OpenStandardError()) { AutoFlush = true };
            Console.SetOut(stdout);
            Console.SetError(stderr);
        }
        catch
        {
            // 已有控制台时可能失败，仍可尝试直接 WriteLine
        }
    }

    private sealed class ForwardingProgress(IProgress<string>? uiProgress) : IProgress<string>
    {
        public void Report(string value)
        {
            WriteLine(value);
            uiProgress?.Report(value);
        }
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetConsoleWindow();

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AllocConsole();

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetConsoleTitle(string lpConsoleTitle);
}
