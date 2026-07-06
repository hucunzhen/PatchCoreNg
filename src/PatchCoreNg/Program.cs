namespace PatchCoreNg;

internal static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return 1;
        }

        var command = args[0].ToLowerInvariant();
        var options = ParseOptions(args.Skip(1).ToArray());

        try
        {
            return command switch
            {
                "train" => RunTrain(options),
                "predict" or "infer" => RunPredict(options),
                "export-help" => RunExportHelp(),
                _ => UnknownCommand(command)
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"错误: {ex.Message}");
            return 1;
        }
    }

    private static int RunTrain(Dictionary<string, string> options)
    {
        var dataPath = Require(options, "data");
        var output = options.GetValueOrDefault("output", "models");
        var profile = options.GetValueOrDefault("profile", "default");
        var config = BuildConfig(options);
        var ngData = options.GetValueOrDefault("ng-data", string.Empty);
        var splitSeed = int.Parse(options.GetValueOrDefault("split-seed", "42"));
        var splitMode = options.GetValueOrDefault("split-mode", "ratio").Equals("count", StringComparison.OrdinalIgnoreCase)
            ? DatasetSplitMode.Count
            : DatasetSplitMode.Ratio;
        var splitOptions = splitMode == DatasetSplitMode.Count
            ? new DatasetSplitOptions
            {
                Mode = DatasetSplitMode.Count,
                OkMemoryCount = int.Parse(options.GetValueOrDefault("ok-memory-count", "6")),
                OkTuneCount = int.Parse(options.GetValueOrDefault("ok-tune-count", "2")),
                NgTuneCount = int.Parse(options.GetValueOrDefault("ng-tune-count", "3")),
                SplitSeed = splitSeed
            }
            : new DatasetSplitOptions
            {
                Mode = DatasetSplitMode.Ratio,
                OkTrainRatio = double.Parse(options.GetValueOrDefault("ok-train-ratio", "0.6")),
                OkTuneRatio = double.Parse(options.GetValueOrDefault("ok-tune-ratio", "0.2")),
                OkTestRatio = double.Parse(options.GetValueOrDefault("ok-test-ratio", "0.2")),
                NgTuneRatio = double.Parse(options.GetValueOrDefault("ng-tune-ratio", "0.5")),
                SplitSeed = splitSeed
            };
        var autoSearchNeighbors = !options.TryGetValue("auto-search-neighbors", out var autoSearch)
            || !string.Equals(autoSearch, "false", StringComparison.OrdinalIgnoreCase);

        Console.WriteLine("=== PatchCore-NG 训练 ===");
        Console.WriteLine($"OK 数据: {dataPath}");
        if (!string.IsNullOrWhiteSpace(ngData))
            Console.WriteLine($"NG 数据: {ngData}");
        Console.WriteLine($"输出目录: {ProfileOutputLayout.GetProfileDirectory(output, profile)}");
        Console.WriteLine($"Coreset: {config.CoresetRatio:P0}, kNN: {config.NumNeighbors}");
        if (splitMode == DatasetSplitMode.Count)
            Console.WriteLine($"划分: 固定数量 OK Memory={splitOptions.OkMemoryCount}, OK 调参={splitOptions.OkTuneCount}, NG 调参={splitOptions.NgTuneCount}, seed={splitSeed}");
        else
            Console.WriteLine($"划分: 比例 OK {splitOptions.OkTrainRatio:P0}/{splitOptions.OkTuneRatio:P0}/{splitOptions.OkTestRatio:P0}, NG 调参 {splitOptions.NgTuneRatio:P0}, seed={splitSeed}");

        var service = new PatchCoreService();
        var progress = new Progress<string>(msg => Console.WriteLine(msg));
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var result = service.TrainAndTune(new TrainAndTuneRequest
        {
            OkDataPath = dataPath,
            NgDataPath = string.IsNullOrWhiteSpace(ngData) ? null : ngData,
            ModelOutputDir = output,
            ProfileName = profile,
            Config = config,
            AutoSearchNeighbors = autoSearchNeighbors,
            SplitOptions = splitOptions
        }, progress);
        stopwatch.Stop();

        Console.WriteLine($"训练完成，模型已保存: {result.ModelPath}");
        ProfileOutputLayout.SaveProfileConfig(output, profile, new PatchCoreSettings
        {
            ProfileName = profile,
            ModelOutputDir = output,
            BackboneId = config.BackboneId,
            NumNeighbors = result.NumNeighbors,
            AnomalyThreshold = result.Threshold
        });
        Console.WriteLine($"Memory Bank={result.MemoryBankSize}, 阈值={result.Threshold:F4}, kNN={result.NumNeighbors}");
        Console.WriteLine($"耗时: {stopwatch.Elapsed.TotalSeconds:F2} 秒");
        return 0;
    }

    private static int RunPredict(Dictionary<string, string> options)
    {
        var modelPath = Require(options, "model");
        var input = Require(options, "input");
        var outputDir = options.GetValueOrDefault("output", "output/predictions");
        var config = BuildConfig(options);

        Console.WriteLine("=== PatchCore-NG 推理 ===");
        Console.WriteLine($"模型: {modelPath}");
        Console.WriteLine($"输入: {input}");
        Console.WriteLine($"热力图: {(config.SaveHeatmap ? "开" : "关")}");
        Console.WriteLine(
            $"kNN: 距离={config.DistanceMetric}, SIMD={config.UseSimdDistance}, Patch并行={config.PatchScoreParallelism}, 降采样={config.FeatureMapDownscale}x, ANN={config.UseApproximateNearestNeighbors}");

        var model = PatchCoreModel.Load(modelPath);
        var inferenceConfig = InferenceConfig.MergeForInference(model, config, modelPath);
        using var predictor = new PatchCorePredictor(model, inferenceConfig);

        var imagePaths = ImagePreprocessor.EnumerateImages(input).ToList();
        if (imagePaths.Count == 0)
            throw new InvalidOperationException($"未找到图像: {input}");

        string? heatmapDir = inferenceConfig.SaveHeatmap ? outputDir : null;
        if (heatmapDir is not null)
            Directory.CreateDirectory(heatmapDir);

        var anomalies = 0;
        var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();

        foreach (var imagePath in imagePaths)
        {
            var result = predictor.Predict(imagePath, heatmapDir);
            Console.WriteLine(
                $"{Path.GetFileName(result.ImagePath)}  score={result.AnomalyScore:F4}  label={result.Label}");
            if (result.StageTiming is { } timing)
            {
                Console.WriteLine($"  {timing.FormatStages()}");
                Console.WriteLine($"  {timing.FormatPercentages()}");
            }

            if (result.IsAnomaly)
                anomalies++;
        }

        totalStopwatch.Stop();
        Console.WriteLine($"完成: {imagePaths.Count} 张, NG={anomalies}, OK={imagePaths.Count - anomalies}");
        Console.WriteLine($"总耗时: {totalStopwatch.Elapsed.TotalSeconds:F2} 秒");
        if (heatmapDir is not null)
            Console.WriteLine($"热力图目录: {heatmapDir}");
        return 0;
    }

    private static int RunExportHelp()
    {
        Console.WriteLine("导出 WideResNet50 等 backbone ONNX:");
        Console.WriteLine("  python scripts/export_backbone.py --list");
        Console.WriteLine("  python scripts/export_backbone.py --backbone wide_resnet50_2");
        Console.WriteLine("  python scripts/export_backbone.py --all");
        return 0;
    }

    private static PatchCoreConfig BuildConfig(Dictionary<string, string> options)
    {
        return new PatchCoreConfig
        {
            BackboneId = options.GetValueOrDefault("backbone-id", BackboneCatalog.DefaultId),
            CustomBackboneOnnxPath = options.GetValueOrDefault("custom-backbone", string.Empty),
            BackboneOnnxPath = ResolveBackbonePath(options),
            ImageSize = int.Parse(options.GetValueOrDefault("image-size", "224")),
            PatchSize = int.Parse(options.GetValueOrDefault("patch-size", "3")),
            NumNeighbors = int.Parse(options.GetValueOrDefault("neighbors", "9")),
            CoresetRatio = double.Parse(options.GetValueOrDefault("coreset", "0.1")),
            TargetEmbedDimension = int.Parse(options.GetValueOrDefault("embed-dim", "1024")),
            AnomalyThreshold = float.Parse(options.GetValueOrDefault("threshold", "0.5")),
            SaveHeatmap = !options.TryGetValue("no-heatmap", out var noHeatmap)
                || string.Equals(noHeatmap, "false", StringComparison.OrdinalIgnoreCase),
            DistanceMetric = ParseDistanceMetric(options.GetValueOrDefault("distance-metric", "SquaredEuclidean")),
            UseSimdDistance = !options.TryGetValue("no-simd", out var noSimd)
                || string.Equals(noSimd, "false", StringComparison.OrdinalIgnoreCase),
            PatchScoreParallelism = int.Parse(options.GetValueOrDefault("patch-parallelism", "0")),
            FeatureMapDownscale = int.Parse(options.GetValueOrDefault("feature-downscale", "1")),
            UseApproximateNearestNeighbors = options.ContainsKey("ann"),
            AnnClusterCount = int.Parse(options.GetValueOrDefault("ann-clusters", "32")),
            AnnProbeClusters = int.Parse(options.GetValueOrDefault("ann-probes", "4")),
        };
    }

    private static DistanceMetric ParseDistanceMetric(string value) =>
        Enum.TryParse<DistanceMetric>(value, ignoreCase: true, out var metric)
            ? metric
            : throw new ArgumentException($"无效距离度量: {value}，可选 Euclidean / SquaredEuclidean");

    private static Dictionary<string, string> ParseOptions(string[] args)
    {
        var options = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--"))
                continue;

            var key = args[i][2..];
            if (i + 1 < args.Length && !args[i + 1].StartsWith("--"))
            {
                options[key] = args[++i];
            }
            else
            {
                options[key] = "true";
            }
        }

        return options;
    }

    private static string Require(Dictionary<string, string> options, string key)
    {
        if (!options.TryGetValue(key, out var value) || string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"缺少参数 --{key}");
        return value;
    }

    private static int UnknownCommand(string command)
    {
        Console.Error.WriteLine($"未知命令: {command}");
        PrintUsage();
        return 1;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("""
            PatchCore-NG - 工业异常检测 (C#)

            用法:
              PatchCoreNg train  --data <OK目录> [--ng-data <NG目录>] [--output models] [--profile default]
              PatchCoreNg predict --model <模型.json> --input <图像或目录> [--output output/predictions]

            可选参数:
              --ng-data           NG 样本目录（用于调参）
              --split-mode        划分方式 ratio 或 count (默认 ratio)
              --ok-train-ratio    OK Memory 比例 (ratio 模式, 默认 0.6)
              --ok-tune-ratio     OK 调参比例 (ratio 模式, 默认 0.2)
              --ok-test-ratio     OK 测试比例 (ratio 模式, 默认 0.2)
              --ng-tune-ratio     NG 调参比例 (ratio 模式, 默认 0.5)
              --ok-memory-count   OK Memory 张数 (count 模式, 默认 6)
              --ok-tune-count     OK 调参张数 (count 模式, 默认 2)
              --ng-tune-count     NG 调参张数 (count 模式, 默认 3)
              --split-seed        随机划分种子 (默认 42)
              --auto-search-neighbors  自动搜索 kNN (默认 true)
              --backbone-id  Backbone 标识 (wide_resnet50_2 / resnet18 / ...)
              --backbone     直接指定 ONNX 路径 (等同 custom)
              --image-size 输入尺寸 (默认 224)
              --patch-size 局部聚合窗口 (默认 3)
              --neighbors  kNN 邻居数 (默认 9)
              --coreset    coreset 采样比例 (默认 0.1)
              --threshold  异常阈值 (默认 0.5)
              --no-heatmap 仅输出分数与判定，不生成热力图
              --distance-metric  距离度量 Euclidean / SquaredEuclidean (默认 SquaredEuclidean)
              --no-simd      禁用 SIMD 向量化距离 (默认启用)
              --patch-parallelism  单图 patch kNN 并行度，0=自动 (默认 0)
              --feature-downscale  特征图降采样倍数 1~8 (默认 1，不降采样)
              --ann          启用近似最近邻索引 (聚类探测)
              --ann-clusters ANN 聚类数 (默认 32)
              --ann-probes   ANN 探测簇数 (默认 4)

            首次使用:
              python scripts/export_backbone.py --all
            """);
    }

    private static string ResolveBackbonePath(Dictionary<string, string> options)
    {
        if (options.TryGetValue("backbone", out var directPath) && !string.IsNullOrWhiteSpace(directPath))
            return AppPaths.Resolve(directPath);

        var backboneId = options.GetValueOrDefault("backbone-id", BackboneCatalog.DefaultId);
        return BackboneCatalog.ResolveOnnxPath(backboneId);
    }
}
