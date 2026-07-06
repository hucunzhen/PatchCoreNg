namespace PatchCoreNg;

public static class AppPaths
{
    public static string ProjectRoot
    {
        get
        {
            var dir = AppContext.BaseDirectory;
            while (!string.IsNullOrEmpty(dir))
            {
                if (File.Exists(Path.Combine(dir, "PatchCoreNg.slnx"))
                    || File.Exists(Path.Combine(dir, "PatchCoreNg.sln"))
                    || Directory.Exists(Path.Combine(dir, "models")))
                {
                    return dir;
                }

                dir = Directory.GetParent(dir)?.FullName ?? string.Empty;
            }

            return AppContext.BaseDirectory;
        }
    }

    public static string Resolve(string path)
    {
        if (Path.IsPathRooted(path))
            return Path.GetFullPath(path);

        return Path.GetFullPath(Path.Combine(ProjectRoot, path));
    }

    public static PatchCoreConfig Resolve(PatchCoreConfig config)
    {
        return new PatchCoreConfig
        {
            BackboneId = config.BackboneId,
            CustomBackboneOnnxPath = config.CustomBackboneOnnxPath,
            BackboneOnnxPath = Resolve(config.BackboneOnnxPath),
            ImageSize = config.ImageSize,
            PatchSize = config.PatchSize,
            NumNeighbors = config.NumNeighbors,
            CoresetRatio = config.CoresetRatio,
            TargetEmbedDimension = config.TargetEmbedDimension,
            AnomalyThreshold = config.AnomalyThreshold,
            UseManualThreshold = config.UseManualThreshold,
            UseGpu = config.UseGpu,
            GpuDeviceId = config.GpuDeviceId,
            InferenceBatchSize = config.InferenceBatchSize,
            PreprocessParallelism = config.PreprocessParallelism,
            SaveHeatmap = config.SaveHeatmap,
        };
    }
}
