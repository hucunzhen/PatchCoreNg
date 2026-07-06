namespace PatchCoreNg;

public sealed class PredictionStageTiming
{
    public TimeSpan Preprocess { get; init; }
    public TimeSpan FeatureExtract { get; init; }
    public TimeSpan FeatureDownscale { get; init; }
    public TimeSpan PatchAggregate { get; init; }
    public TimeSpan KnnsScore { get; init; }
    public TimeSpan Judgment { get; init; }
    public TimeSpan Heatmap { get; init; }

    public TimeSpan Scoring => FeatureDownscale + PatchAggregate + KnnsScore;

    public TimeSpan Total =>
        Preprocess + FeatureExtract + Scoring + Judgment + Heatmap;

    public string FormatStages() =>
        $"预处理 {Format(Preprocess)} | 特征提取 {Format(FeatureExtract)} | " +
        $"降采样 {Format(FeatureDownscale)} | Patch聚合 {Format(PatchAggregate)} | " +
        $"kNN {Format(KnnsScore)} | 判定 {Format(Judgment)} | 热力图 {Format(Heatmap)} | " +
        $"合计 {Format(Total)}";

    public string FormatPercentages()
    {
        var totalMs = Total.TotalMilliseconds;
        if (totalMs <= 0)
            return string.Empty;

        static string Pct(TimeSpan stage, double total) =>
            total > 0 ? $"{stage.TotalMilliseconds / total * 100:F1}%" : "0.0%";

        return
            $"占比: 预处理 {Pct(Preprocess, totalMs)} | 特征提取 {Pct(FeatureExtract, totalMs)} | " +
            $"降采样 {Pct(FeatureDownscale, totalMs)} | Patch聚合 {Pct(PatchAggregate, totalMs)} | " +
            $"kNN {Pct(KnnsScore, totalMs)} | 判定 {Pct(Judgment, totalMs)} | 热力图 {Pct(Heatmap, totalMs)}";
    }

    public string FormatDetailed() => $"{FormatStages()}{Environment.NewLine}  {FormatPercentages()}";

    public static string Format(TimeSpan elapsed) => StepProgress.FormatElapsed(elapsed);
}
