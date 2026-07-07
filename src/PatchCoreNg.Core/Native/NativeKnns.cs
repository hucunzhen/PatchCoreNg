using System.Runtime.InteropServices;

namespace PatchCoreNg;

internal static partial class NativeKnns
{
    private const string LibName = "patchcore_native";
    private static readonly bool Available = TryLoadNative();

    public static bool IsAvailable => Available;

    public static string? LoadError { get; } = Available ? null : BuildLoadError();

    private static bool TryLoadNative()
    {
        var assemblyDir = Path.GetDirectoryName(typeof(NativeKnns).Assembly.Location);
        if (!string.IsNullOrEmpty(assemblyDir))
        {
            var path = Path.Combine(assemblyDir, $"{LibName}.dll");
            if (File.Exists(path) && NativeLibrary.TryLoad(path, out _))
                return true;
        }

        if (NativeLibrary.TryLoad(LibName, typeof(NativeKnns).Assembly, DllImportSearchPath.AssemblyDirectory, out _))
            return true;

        if (NativeLibrary.TryLoad(LibName, typeof(NativeKnns).Assembly, DllImportSearchPath.ApplicationDirectory, out _))
            return true;

        return false;
    }

    private static string BuildLoadError()
    {
        var assemblyDir = Path.GetDirectoryName(typeof(NativeKnns).Assembly.Location) ?? "(unknown)";
        var appDir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return $"patchcore_native.dll 未找到或无法加载。请将 DLL 放到程序目录，或运行 scripts/build_native.ps1 后重新编译。已检查: {assemblyDir}; {appDir}";
    }

    public static bool TryCreateBank(
        float[][] embeddings,
        KnnsSearchOptions options,
        out NativeMemoryBankHandle? handle)
    {
        handle = null;
        if (!IsAvailable || embeddings.Length == 0)
            return false;

        var dim = embeddings[0].Length;
        var flat = new float[embeddings.Length * dim];
        for (var i = 0; i < embeddings.Length; i++)
        {
            if (embeddings[i].Length != dim)
                return false;
            embeddings[i].AsSpan().CopyTo(flat.AsSpan(i * dim, dim));
        }

        var config = new PcnBankConfig
        {
            use_ann = options.UseApproximateNearestNeighbors ? 1 : 0,
            ann_cluster_count = options.AnnClusterCount,
            ann_probe_clusters = options.AnnProbeClusters,
            distance_metric = (int)options.DistanceMetric,
            use_simd = options.UseSimdDistance ? 1 : 0,
        };

        var ptr = pcn_bank_create(flat, embeddings.Length, dim, ref config);
        if (ptr == IntPtr.Zero)
            return false;

        handle = new NativeMemoryBankHandle(ptr);
        return true;
    }

    public static bool TryAggregateAndScore(
        NativeMemoryBankHandle bank,
        FeatureMap map,
        int patchSize,
        int numNeighbors,
        int patchParallelism,
        float[] outDistances,
        out float imageScore)
    {
        imageScore = 0f;
        if (!IsAvailable)
            return false;

        var expected = map.Height * map.Width;
        if (outDistances.Length != expected)
            throw new ArgumentException($"输出分数长度应为 {expected}。");

        var scoreParams = new PcnScoreParams
        {
            patch_size = patchSize,
            num_neighbors = numNeighbors,
            patch_parallelism = patchParallelism,
        };

        var rc = pcn_aggregate_and_score(
            bank.Handle,
            map.Data,
            map.Channels,
            map.Height,
            map.Width,
            ref scoreParams,
            outDistances,
            out imageScore);

        if (rc != 0)
        {
            var errPtr = pcn_last_error();
            var err = errPtr != IntPtr.Zero ? Marshal.PtrToStringUTF8(errPtr) : null;
            throw new InvalidOperationException(
                string.IsNullOrWhiteSpace(err) ? $"Native kNN failed ({rc})" : err);
        }

        return true;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PcnBankConfig
    {
        public int use_ann;
        public int ann_cluster_count;
        public int ann_probe_clusters;
        public int distance_metric;
        public int use_simd;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PcnScoreParams
    {
        public int patch_size;
        public int num_neighbors;
        public int patch_parallelism;
    }

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr pcn_bank_create(
        float[] embeddings,
        int bank_count,
        int dim,
        ref PcnBankConfig config);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void pcn_bank_destroy(IntPtr bank);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int pcn_aggregate_and_score(
        IntPtr bank,
        float[] feature_chw,
        int channels,
        int height,
        int width,
        ref PcnScoreParams score_params,
        float[] out_distances,
        out float out_image_score);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr pcn_last_error();

    internal sealed class NativeMemoryBankHandle : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public NativeMemoryBankHandle(IntPtr handle) => _handle = handle;

        public IntPtr Handle
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(NativeMemoryBankHandle));
                return _handle;
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            if (_handle != IntPtr.Zero)
            {
                pcn_bank_destroy(_handle);
                _handle = IntPtr.Zero;
            }

            _disposed = true;
        }
    }
}
