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
    public TuningMetrics? TestMetrics { get; init; }
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
    public required string OkDataPath { get; init; }
    public string? NgDataPath { get; init; }
    public required string ModelOutputDir { get; init; }
    public required string ProfileName { get; init; }
    public required PatchCoreConfig Config { get; init; }
    public bool AutoSearchNeighbors { get; init; } = true;
    public DatasetSplitOptions SplitOptions { get; init; } = new();
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
            OkDataPath = trainDataPath,
            ModelOutputDir = ProfileOutputLayout.NormalizeOutputBaseDir(modelOutputPath),
            ProfileName = "default",
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
        var resolvedOkPath = AppPaths.Resolve(request.OkDataPath);
        ProfileOutputLayout.EnsureProfileDirectory(request.ModelOutputDir, request.ProfileName);
        var resolvedOutput = AppPaths.Resolve(
            ProfileOutputLayout.GetModelPath(request.ModelOutputDir, request.ProfileName));

        var splitOptions = request.SplitOptions;
        var split = DatasetSplitter.Split(
            resolvedOkPath,
            string.IsNullOrWhiteSpace(request.NgDataPath) ? null : AppPaths.Resolve(request.NgDataPath),
            splitOptions);

        progress?.Report(
            $"数据集划分 ({(splitOptions.Mode == DatasetSplitMode.Count ? "固定数量" : "比例")}, seed={splitOptions.SplitSeed}): " +
            $"OK Memory={split.OkTrainPaths.Count}, OK 调参={split.OkTunePaths.Count}, OK 测试={split.OkTestPaths.Count}, " +
            $"NG 调参={split.NgTunePaths.Count}, NG 测试={split.NgTestPaths.Count}");

        if (splitOptions.Mode == DatasetSplitMode.Count)
        {
            progress?.Report(
                $"固定数量: OK Memory={splitOptions.OkMemoryCount}, OK 调参={splitOptions.OkTuneCount}, " +
                $"NG 调参={splitOptions.NgTuneCount}（其余为测试）");
        }
        else
        {
            progress?.Report(
                $"OK 比例 Memory/调参/测试={splitOptions.OkTrainRatio:P0}/{splitOptions.OkTuneRatio:P0}/{splitOptions.OkTestRatio:P0}, " +
                $"NG 调参比例={splitOptions.NgTuneRatio:P0}");
        }

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        using var trainer = new PatchCoreTrainer(resolvedConfig);
        var model = trainer.Train(split.OkTrainPaths, progress);

        TuningMetrics? tuning = null;
        var finalThreshold = model.AnomalyThreshold;
        var finalNeighbors = model.NumNeighbors;

        if (split.CanTune)
        {
            progress?.Report($"调参: OK={split.OkTunePaths.Count} 张, NG={split.NgTunePaths.Count} 张");

            tuning = request.AutoSearchNeighbors
                ? ParameterTuner.TuneNeighbors(
                    model,
                    resolvedConfig,
                    split.OkTunePaths,
                    split.NgTunePaths,
                    progress: progress)
                : ParameterTuner.Tune(
                    model,
                    resolvedConfig,
                    ParameterTuner.ScoreImages(model, resolvedConfig, resolvedConfig.NumNeighbors, split.OkTunePaths),
                    ParameterTuner.ScoreImages(model, resolvedConfig, resolvedConfig.NumNeighbors, split.NgTunePaths),
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
        else if (split.NgTunePaths.Count == 0 && split.NgTestPaths.Count == 0)
        {
            progress?.Report("未提供 NG 目录或 NG 样本为空，跳过 OK/NG 调参，使用训练集 P95 阈值。");
        }
        else
        {
            progress?.Report("调参样本不足（OK 或 NG 调参集为空），跳过 OK/NG 调参，使用训练集 P95 阈值。");
        }

        model = model.WithTunedParams(finalThreshold, finalNeighbors);

        TuningMetrics? testMetrics = null;
        if (split.CanTest)
        {
            progress?.Report($"测试集评估: OK={split.OkTestPaths.Count} 张, NG={split.NgTestPaths.Count} 张");
            testMetrics = ParameterTuner.EvaluateAtThreshold(
                model,
                resolvedConfig,
                finalNeighbors,
                split.OkTestPaths,
                split.NgTestPaths,
                finalThreshold);
            progress?.Report(
                $"测试结果: F1={testMetrics.F1:P1}, Acc={testMetrics.Accuracy:P1}, " +
                $"Prec={testMetrics.Precision:P1}, Rec={testMetrics.Recall:P1}");
            progress?.Report(
                $"测试混淆矩阵: TP={testMetrics.TruePositive}, TN={testMetrics.TrueNegative}, " +
                $"FP={testMetrics.FalsePositive}, FN={testMetrics.FalseNegative}");
        }
        else if (split.OkTestPaths.Count > 0 || split.NgTestPaths.Count > 0)
        {
            progress?.Report("测试集不完整（需同时有 OK 测试与 NG 测试样本），跳过测试评估。");
        }

        model.Save(resolvedOutput);
        stopwatch.Stop();

        return new TrainResult
        {
            ModelPath = resolvedOutput,
            ImageCount = split.OkTrainPaths.Count,
            MemoryBankSize = model.MemoryBank.Length,
            Threshold = finalThreshold,
            NumNeighbors = finalNeighbors,
            Elapsed = stopwatch.Elapsed,
            TuningMetrics = tuning,
            TestMetrics = testMetrics
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
