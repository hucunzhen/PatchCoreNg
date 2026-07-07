namespace PatchCoreNg;

public static class InferenceConfig
{
    /// <summary>
    /// 推理时以模型内保存的参数为准，避免 UI 与训练配置不一致导致分数异常。
    /// </summary>
    public static PatchCoreConfig MergeForInference(
        PatchCoreModel model,
        PatchCoreConfig? userConfig,
        string? modelFilePath = null)
    {
        userConfig ??= new PatchCoreConfig();
        var backboneId = string.IsNullOrWhiteSpace(model.BackboneId)
            ? userConfig.BackboneId
            : model.BackboneId;
        var customBackbone = userConfig.CustomBackboneOnnxPath;

        if (!string.IsNullOrWhiteSpace(modelFilePath))
        {
            var profileConfigPath = Path.Combine(
                Path.GetDirectoryName(Path.GetFullPath(modelFilePath))!,
                ProfileOutputLayout.ConfigFileName);
            if (File.Exists(profileConfigPath))
            {
                var profile = SettingsStore.Load(profileConfigPath);
                if (string.IsNullOrWhiteSpace(model.BackboneId))
                    backboneId = profile.BackboneId;
                if (string.IsNullOrWhiteSpace(customBackbone))
                    customBackbone = profile.CustomBackboneOnnxPath;
            }
        }

        return new PatchCoreConfig
        {
            BackboneId = backboneId,
            CustomBackboneOnnxPath = customBackbone,
            BackboneOnnxPath = AppPaths.Resolve(
                BackboneCatalog.ResolveOnnxPath(backboneId, customBackbone, userConfig.UseGpu)),
            ImageSize = model.ImageSize,
            PatchSize = model.PatchSize,
            NumNeighbors = model.NumNeighbors,
            CoresetRatio = model.CoresetRatio,
            TargetEmbedDimension = model.TargetEmbedDimension,
            AnomalyThreshold = userConfig.UseManualThreshold
                ? userConfig.AnomalyThreshold
                : model.AnomalyThreshold,
            UseManualThreshold = userConfig.UseManualThreshold,
            UseGpu = userConfig.UseGpu,
            GpuDeviceId = userConfig.GpuDeviceId,
            InferenceBatchSize = userConfig.InferenceBatchSize,
            PreprocessParallelism = userConfig.PreprocessParallelism,
            SaveHeatmap = userConfig.SaveHeatmap,
            UseApproximateNearestNeighbors = model.UseApproximateNearestNeighbors,
            AnnClusterCount = model.AnnClusterCount,
            AnnProbeClusters = model.AnnProbeClusters,
            DistanceMetric = model.DistanceMetric,
            UseSimdDistance = model.UseSimdDistance,
            PatchScoreParallelism = model.PatchScoreParallelism,
            FeatureMapDownscale = model.FeatureMapDownscale,
        };
    }
}
