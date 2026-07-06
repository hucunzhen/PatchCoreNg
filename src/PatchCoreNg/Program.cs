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
        var output = options.GetValueOrDefault("output", "models/patchcore_model.json");
        var config = BuildConfig(options);

        Console.WriteLine("=== PatchCore-NG 训练 ===");
        Console.WriteLine($"数据: {dataPath}");
        Console.WriteLine($"输出: {output}");
        Console.WriteLine($"Coreset: {config.CoresetRatio:P0}, kNN: {config.NumNeighbors}");

        using var trainer = new PatchCoreTrainer(config);
        var progress = new Progress<string>(msg => Console.WriteLine(msg));
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var model = trainer.Train(dataPath, progress);
        model.Save(output);
        stopwatch.Stop();

        Console.WriteLine($"训练完成，模型已保存: {output}");
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

        var model = PatchCoreModel.Load(modelPath);
        using var predictor = new PatchCorePredictor(model, config);

        var imagePaths = ImagePreprocessor.EnumerateImages(input).ToList();
        if (imagePaths.Count == 0)
            throw new InvalidOperationException($"未找到图像: {input}");

        Directory.CreateDirectory(outputDir);
        var anomalies = 0;
        var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();

        foreach (var imagePath in imagePaths)
        {
            var itemStopwatch = System.Diagnostics.Stopwatch.StartNew();
            var result = predictor.Predict(imagePath, outputDir);
            itemStopwatch.Stop();
            Console.WriteLine(
                $"{Path.GetFileName(result.ImagePath)}  score={result.AnomalyScore:F4}  label={result.Label}  耗时={itemStopwatch.Elapsed.TotalSeconds:F2}s");
            if (result.IsAnomaly)
                anomalies++;
        }

        totalStopwatch.Stop();
        Console.WriteLine($"完成: {imagePaths.Count} 张, NG={anomalies}, OK={imagePaths.Count - anomalies}");
        Console.WriteLine($"总耗时: {totalStopwatch.Elapsed.TotalSeconds:F2} 秒");
        Console.WriteLine($"热力图目录: {outputDir}");
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
            AnomalyThreshold = float.Parse(options.GetValueOrDefault("threshold", "0.5"))
        };
    }

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
              PatchCoreNg train  --data <正常样本目录> [--output models/patchcore_model.json]
              PatchCoreNg predict --model <模型.json> --input <图像或目录> [--output output/predictions]

            可选参数:
              --backbone-id  Backbone 标识 (wide_resnet50_2 / resnet18 / ...)
              --backbone     直接指定 ONNX 路径 (等同 custom)
              --image-size 输入尺寸 (默认 224)
              --patch-size 局部聚合窗口 (默认 3)
              --neighbors  kNN 邻居数 (默认 9)
              --coreset    coreset 采样比例 (默认 0.1)
              --threshold  异常阈值 (默认 0.5)

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
