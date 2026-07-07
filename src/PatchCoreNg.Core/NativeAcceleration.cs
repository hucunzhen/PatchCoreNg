namespace PatchCoreNg;

public static class NativeAcceleration
{
    public static bool IsAvailable => NativeKnns.IsAvailable;

    public static string? LoadError => NativeKnns.LoadError;

    public static string DescribeStatus() =>
        IsAvailable ? "Native(C++) 已加载" : $"Managed(C#) 回退 — {LoadError}";
}
