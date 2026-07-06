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

        var log = new StepProgress(progress);

        var resolvedConfig = AppPaths.Resolve(request.Config);

        var resolvedOkPath = AppPaths.Resolve(request.OkDataPath);

        ProfileOutputLayout.EnsureProfileDirectory(request.ModelOutputDir, request.ProfileName);

        var resolvedOutput = AppPaths.Resolve(

            ProfileOutputLayout.GetModelPath(request.ModelOutputDir, request.ProfileName));



        var split = log.Run(

            "数据集划分",

            () => DatasetSplitter.Split(

                resolvedOkPath,

                string.IsNullOrWhiteSpace(request.NgDataPath) ? null : AppPaths.Resolve(request.NgDataPath),

                request.SplitOptions),

            $"模式={(request.SplitOptions.Mode == DatasetSplitMode.Count ? "固定数量" : "比例")}, seed={request.SplitOptions.SplitSeed}",

            result =>

                $"OK 训练={result.OkTrainPaths.Count}, OK 调参={result.OkTunePaths.Count}, OK 测试={result.OkTestPaths.Count}, " +

                $"NG 调参={result.NgTunePaths.Count}, NG 测试={result.NgTestPaths.Count}");



        if (request.SplitOptions.Mode == DatasetSplitMode.Count)

        {

            log.Info(

                "数据集划分",

                $"固定数量: OK Memory={request.SplitOptions.OkMemoryCount}, OK 调参={request.SplitOptions.OkTuneCount}, " +

                $"NG 调参={request.SplitOptions.NgTuneCount}");

        }

        else

        {

            log.Info(

                "数据集划分",

                $"OK 比例 Memory/调参/测试={request.SplitOptions.OkTrainRatio:P0}/{request.SplitOptions.OkTuneRatio:P0}/{request.SplitOptions.OkTestRatio:P0}, " +

                $"NG 调参比例={request.SplitOptions.NgTuneRatio:P0}");

        }



        cancellationToken.ThrowIfCancellationRequested();

        using var trainer = new PatchCoreTrainer(resolvedConfig);

        var model = log.Run(

            "模型训练",

            () => trainer.Train(split.OkTrainPaths, progress),

            $"{split.OkTrainPaths.Count} 张 OK 训练图",

            trained => $"Memory Bank={trained.MemoryBank.Length}, 阈值={trained.AnomalyThreshold:F4}");



        TuningMetrics? tuning = null;

        var finalThreshold = model.AnomalyThreshold;

        var finalNeighbors = model.NumNeighbors;



        if (split.CanTune)

        {

            tuning = log.Run(

                "OK/NG调参",

                () => request.AutoSearchNeighbors

                    ? ParameterTuner.TuneNeighbors(

                        model,

                        resolvedConfig,

                        split.OkTunePaths,

                        split.NgTunePaths,

                        progress: progress)

                    : ParameterTuner.Tune(

                        model,

                        resolvedConfig,

                        ParameterTuner.ScoreImages(model, resolvedConfig, resolvedConfig.NumNeighbors, split.OkTunePaths, log, "调参-OK"),

                        ParameterTuner.ScoreImages(model, resolvedConfig, resolvedConfig.NumNeighbors, split.NgTunePaths, log, "调参-NG"),

                        resolvedConfig.NumNeighbors),

                $"OK={split.OkTunePaths.Count}, NG={split.NgTunePaths.Count}",

                metrics =>

                    $"阈值={metrics.Threshold:F4}, k={metrics.NumNeighbors}, F1={metrics.F1:P1}, " +

                    $"TP={metrics.TruePositive}, TN={metrics.TrueNegative}, FP={metrics.FalsePositive}, FN={metrics.FalseNegative}");



            finalThreshold = tuning.Threshold;

            finalNeighbors = tuning.NumNeighbors;

        }

        else if (split.NgTunePaths.Count == 0 && split.NgTestPaths.Count == 0)

        {

            log.Info("OK/NG调参", "未提供 NG 样本，跳过调参，使用训练集 P95 阈值");

        }

        else

        {

            log.Info("OK/NG调参", "调参样本不足，跳过调参，使用训练集 P95 阈值");

        }



        model = model.WithTunedParams(finalThreshold, finalNeighbors);



        TuningMetrics? testMetrics = null;

        if (split.CanTest)

        {

            testMetrics = log.Run(

                "测试集评估",

                () => ParameterTuner.EvaluateAtThreshold(

                    model,

                    resolvedConfig,

                    finalNeighbors,

                    split.OkTestPaths,

                    split.NgTestPaths,

                    finalThreshold,

                    progress),

                $"OK={split.OkTestPaths.Count}, NG={split.NgTestPaths.Count}",

                metrics =>

                    $"F1={metrics.F1:P1}, Acc={metrics.Accuracy:P1}, TP={metrics.TruePositive}, TN={metrics.TrueNegative}, FP={metrics.FalsePositive}, FN={metrics.FalseNegative}");

        }

        else if (split.OkTestPaths.Count > 0 || split.NgTestPaths.Count > 0)

        {

            log.Info("测试集评估", "测试集不完整，跳过评估");

        }



        log.Run(
            "保存模型",
            () => model.Save(resolvedOutput),
            resolvedOutput,
            Path.GetFileName(resolvedOutput));



        log.Complete($"训练流程结束 | 模型={resolvedOutput}");



        return new TrainResult

        {

            ModelPath = resolvedOutput,

            ImageCount = split.OkTrainPaths.Count,

            MemoryBankSize = model.MemoryBank.Length,

            Threshold = finalThreshold,

            NumNeighbors = finalNeighbors,

            Elapsed = log.TotalElapsed,

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

        var log = new StepProgress(progress);

        var resolvedModelPath = AppPaths.Resolve(modelPath);

        var resolvedOutputDir = string.IsNullOrWhiteSpace(outputDir)

            ? null

            : AppPaths.Resolve(outputDir);



        var model = log.Run(

            "加载模型",

            () => PatchCoreModel.Load(resolvedModelPath),

            Path.GetFileName(resolvedModelPath),

            loaded => $"Memory Bank={loaded.MemoryBank.Length}, 阈值={loaded.AnomalyThreshold:F4}");



        var resolvedConfig = AppPaths.Resolve(
            InferenceConfig.MergeForInference(model, config, resolvedModelPath));

        var heatmapOutputDir = resolvedConfig.SaveHeatmap ? resolvedOutputDir : null;

        if (config is not null &&
            (config.ImageSize != model.ImageSize ||
             config.PatchSize != model.PatchSize ||
             config.NumNeighbors != model.NumNeighbors))
        {
            log.Info(
                "批量推理",
                $"已使用模型内参数: image={model.ImageSize}, patch={model.PatchSize}, k={model.NumNeighbors}");
        }



        var pathList = inputPaths.ToList();

        using var predictor = new PatchCorePredictor(model, resolvedConfig);

        log.Info(
            "批量推理",
            $"{pathList.Count} 张 | 逐张处理 | 设备={predictor.ExecutionProvider} | kNN并行={FeaturePipeline.ResolvePreprocessParallelism(resolvedConfig.PreprocessParallelism)} | 热力图={(resolvedConfig.SaveHeatmap ? "开" : "关")}");

        if (heatmapOutputDir is not null)
            Directory.CreateDirectory(heatmapOutputDir);

        var items = new List<TimedPredictionResult>(pathList.Count);

        for (var i = 0; i < pathList.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var imagePath = pathList[i];
            var itemStopwatch = System.Diagnostics.Stopwatch.StartNew();
            var result = predictor.Predict(imagePath, heatmapOutputDir);
            itemStopwatch.Stop();

            items.Add(new TimedPredictionResult
            {
                Result = result,
                Elapsed = itemStopwatch.Elapsed,
            });

            log.Info(
                "推理",
                $"{i + 1}/{pathList.Count} {Path.GetFileName(imagePath)} score={result.AnomalyScore:F4} {result.Label} 耗时={StepProgress.FormatElapsed(itemStopwatch.Elapsed)}");
        }

        var ngCount = items.Count(x => x.Result.IsAnomaly);
        log.Complete($"推理结束 | {pathList.Count} 张 | NG={ngCount}, OK={pathList.Count - ngCount}");

        return new PredictBatchResult
        {
            Items = items,
            TotalElapsed = log.TotalElapsed,
        };

    }

}


