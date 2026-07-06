namespace PatchCoreNg;

public sealed class TrainResult
{
    public required string ModelPath { get; init; }
    public required int ImageCount { get; init; }
    public required int MemoryBankSize { get; init; }
    public required float Threshold { get; init; }
    public required int NumNeighbors { get; init; }
    public required TimeSpan Elapsed { get; init; }
    public TuningMetrics? TuningMetrics { get; init; }
}

public sealed class TimedPredictionResult
{
    public required PredictionResult Result { get; init; }
    public required TimeSpan Elapsed { get; init; }
}

public sealed class PredictBatchResult
{
    public required IReadOnlyList<TimedPredictionResult> Items { get; init; }
    public required TimeSpan TotalElapsed { get; init; }
}

public sealed class TrainAndTuneRequest
{
    public required string OkTrainPath { get; init; }
    public string? OkTunePath { get; init; }
    public string? NgTunePath { get; init; }
    public required string ModelOutputPath { get; init; }
    public required PatchCoreConfig Config { get; init; }
    public bool AutoSearchNeighbors { get; init; } = true;
}

public sealed class PatchCoreService
{
    public TrainResult Train(
        string trainDataPath,
        string modelOutputPath,
        PatchCoreConfig config,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return TrainAndTune(new TrainAndTuneRequest
        {
            OkTrainPath = trainDataPath,
            ModelOutputPath = modelOutputPath,
            Config = config
        }, progress, cancellationToken);
    }

    public TrainResult TrainAndTune(
        TrainAndTuneRequest request,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var resolvedConfig = AppPaths.Resolve(request.Config);
        var resolvedTrainPath = AppPaths.Resolve(request.OkTrainPath);
        var resolvedOutput = AppPaths.Resolve(request.ModelOutputPath);

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        using var trainer = new PatchCoreTrainer(resolvedConfig);
        var model = trainer.Train(resolvedTrainPath, progress);

        TuningMetrics? tuning = null;
        var finalThreshold = model.AnomalyThreshold;
        var finalNeighbors = model.NumNeighbors;

        var ngTunePath = string.IsNullOrWhiteSpace(request.NgTunePath)
            ? null
            : AppPaths.Resolve(request.NgTunePath);
        var okTunePath = string.IsNullOrWhiteSpace(request.OkTunePath)
            ? resolvedTrainPath
            : AppPaths.Resolve(request.OkTunePath);

        if (!string.IsNullOrWhiteSpace(ngTunePath) && Directory.Exists(ngTunePath))
        {
            var okTuneImages = ImagePreprocessor.EnumerateImages(okTunePath).ToList();
            var ngTuneImages = ImagePreprocessor.EnumerateImages(ngTunePath).ToList();

            if (okTuneImages.Count == 0 || ngTuneImages.Count == 0)
            {
                progress?.Report("调参样本不足，使用训练集 P95 阈值。");
            }
            else
            {
                progress?.Report($"调参: OK={okTuneImages.Count} 张, NG={ngTuneImages.Count} 张");
                if (okTunePath == resolvedTrainPath)
                    progress?.Report("提示: 未指定 OK 调参目录，使用训练目录评分（建议单独提供验证 OK）。");

                tuning = request.AutoSearchNeighbors
                    ? ParameterTuner.TuneNeighbors(
                        model,
                        resolvedConfig,
                        okTuneImages,
                        ngTuneImages,
                        progress: progress)
                    : ParameterTuner.Tune(
                        model,
                        resolvedConfig,
                        ParameterTuner.ScoreImages(model, resolvedConfig, resolvedConfig.NumNeighbors, okTuneImages),
                        ParameterTuner.ScoreImages(model, resolvedConfig, resolvedConfig.NumNeighbors, ngTuneImages),
                        resolvedConfig.NumNeighbors);

                finalThreshold = tuning.Threshold;
                finalNeighbors = tuning.NumNeighbors;
                progress?.Report(
                    $"调参结果: 阈值={finalThreshold:F4}, kNN={finalNeighbors}, F1={tuning.F1:P1}, " +
                    $"Acc={tuning.Accuracy:P1}, Prec={tuning.Precision:P1}, Rec={tuning.Recall:P1}");
                progress?.Report(
                    $"混淆矩阵: TP={tuning.TruePositive}, TN={tuning.TrueNegative}, " +
                    $"FP={tuning.FalsePositive}, FN={tuning.FalseNegative}");
            }
        }
        else
        {
            progress?.Report("未提供 NG 调参目录，跳过 OK/NG 调参，使用训练集 P95 阈值。");
        }

        model = model.WithTunedParams(finalThreshold, finalNeighbors);
        model.Save(resolvedOutput);
        stopwatch.Stop();

        return new TrainResult
        {
            ModelPath = resolvedOutput,
            ImageCount = ImagePreprocessor.EnumerateImages(resolvedTrainPath).Count(),
            MemoryBankSize = model.MemoryBank.Length,
            Threshold = finalThreshold,
            NumNeighbors = finalNeighbors,
            Elapsed = stopwatch.Elapsed,
            TuningMetrics = tuning
        };
    }

    public PredictBatchResult Predict(
        string modelPath,
        IEnumerable<string> inputPaths,
        string? outputDir,
        PatchCoreConfig? config = null,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var resolvedModelPath = AppPaths.Resolve(modelPath);
        var resolvedOutputDir = string.IsNullOrWhiteSpace(outputDir)
            ? null
            : AppPaths.Resolve(outputDir);

        var model = PatchCoreModel.Load(resolvedModelPath);
        var resolvedConfig = AppPaths.Resolve(config ?? new PatchCoreConfig
        {
            ImageSize = model.ImageSize,
            PatchSize = model.PatchSize,
            NumNeighbors = model.NumNeighbors,
            CoresetRatio = model.CoresetRatio,
            TargetEmbedDimension = model.TargetEmbedDimension,
            AnomalyThreshold = model.AnomalyThreshold
        });

        var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
        var items = new List<TimedPredictionResult>();

        using var predictor = new PatchCorePredictor(model, resolvedConfig);
        foreach (var path in inputPaths)
        {
            cancellationToken.ThrowIfCancellationRequested();
            progress?.Report($"推理: {Path.GetFileName(path)}");

            var itemStopwatch = System.Diagnostics.Stopwatch.StartNew();
            var result = predictor.Predict(path, resolvedOutputDir);
            itemStopwatch.Stop();

            items.Add(new TimedPredictionResult
            {
                Result = result,
                Elapsed = itemStopwatch.Elapsed
            });
        }

        totalStopwatch.Stop();
        return new PredictBatchResult
        {
            Items = items,
            TotalElapsed = totalStopwatch.Elapsed
        };
    }
}
