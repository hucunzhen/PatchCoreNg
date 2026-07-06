using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using Microsoft.Win32;
using PatchCoreNg;

namespace PatchCoreNg.App;

public sealed class PredictionRowViewModel
{
    public required string FileName { get; init; }
    public required string ImagePath { get; init; }
    public string? HeatmapPath { get; init; }
    public required float Score { get; init; }
    public required string Label { get; init; }
    public required string ElapsedText { get; init; }
    public string TimingDetail { get; init; } = string.Empty;
}

public sealed class MainViewModel : INotifyPropertyChanged
{
    private readonly PatchCoreService _service = new();
    private readonly PatchCoreSettings _settings = PatchCoreSettings.CreateDefault();

    private string _okDataPath = string.Empty;
    private string _ngDataPath = string.Empty;
    private bool _autoSearchNeighbors = true;
    private string _tuneResultText = string.Empty;
    private string _testResultText = string.Empty;
    private string _modelOutputDir = "models";
    private string _trainLog = string.Empty;
    private string _trainTimeText = "耗时: -";
    private string _modelPath = string.Empty;
    private string _inferInputPath = string.Empty;
    private string _inferOutputPath = AppPaths.Resolve("output/predictions");
    private bool _inferSingleImage = true;
    private string _inferTimeText = "总耗时: -";
    private string _inferSummary = string.Empty;
    private string _inferTimingDetail = string.Empty;
    private string? _previewOriginalPath;
    private string? _previewHeatmapPath;
    private bool _isBusy;

    private string _paramsConfigDir = SettingsStore.DefaultConfigDirectory;
    private string _profileName = "default";
    private string _selectedProfile = "default";
    private string _activeProfileName = "default";
    private bool _isProfileDirty;
    private bool _suppressProfileSwitch;
    private bool _suppressDirty;
    private string _paramsStatus = string.Empty;
    private string _selectedBackboneId = BackboneCatalog.DefaultId;
    private string _backboneStatus = string.Empty;
    private string _backboneDescription = string.Empty;
    private string _backboneExportHint = string.Empty;

    public MainViewModel()
    {
        foreach (var backbone in BackboneCatalog.All)
            AvailableBackbones.Add(backbone);

        BrowseOkDataCommand = new RelayCommand(_ => BrowseOkData());
        BrowseNgDataCommand = new RelayCommand(_ => BrowseNgData());
        BrowseModelOutputCommand = new RelayCommand(_ => BrowseModelOutput());
        TrainCommand = new AsyncRelayCommand(_ => TrainAsync(), _ => CanRunTrain());

        BrowseModelCommand = new RelayCommand(_ => BrowseModel());
        BrowseInferInputCommand = new RelayCommand(_ => BrowseInferInput());
        BrowseInferOutputCommand = new RelayCommand(_ => BrowseInferOutput());
        PredictCommand = new AsyncRelayCommand(_ => PredictAsync(), _ => CanRunPredict());

        BrowseParamsConfigDirCommand = new RelayCommand(_ => BrowseParamsConfigDir());
        BrowseBackboneCommand = new RelayCommand(_ => BrowseBackbone(), _ => IsCustomBackbone);
        SaveParamsCommand = new RelayCommand(_ => SaveCurrentProfile(), _ => !IsBusy && !string.IsNullOrWhiteSpace(SelectedProfile));
        SaveProfileAsCommand = new RelayCommand(_ => SaveProfileAs(), _ => !IsBusy);
        NewProfileCommand = new RelayCommand(_ => NewProfile(), _ => !IsBusy);
        DeleteProfileCommand = new RelayCommand(_ => DeleteProfile(), _ => !IsBusy && !string.IsNullOrWhiteSpace(SelectedProfile));
        ResetParamsCommand = new RelayCommand(_ => ResetParams(), _ => !IsBusy);
        RefreshProfilesCommand = new RelayCommand(_ => RefreshProfiles());

        EnsureDefaultConfig();
        RefreshProfiles();
        LoadStartupProfile();
        SyncModelPathsFromProfile();
        UpdateBackboneUi();
    }

    private static readonly HashSet<string> DirtyProperties =
    [
        nameof(OkDataPath),
        nameof(NgDataPath),
        nameof(OkTrainRatio),
        nameof(OkTuneRatio),
        nameof(OkTestRatio),
        nameof(NgTuneRatio),
        nameof(UseRatioSplit),
        nameof(UseCountSplit),
        nameof(OkMemoryCount),
        nameof(OkTuneCount),
        nameof(NgTuneCount),
        nameof(SplitSeed),
        nameof(ModelOutputDir),
        nameof(ModelPath),
        nameof(AutoSearchNeighbors),
        nameof(SelectedBackboneId),
        nameof(BackboneOnnxPath),
        nameof(ImageSize),
        nameof(PatchSize),
        nameof(NumNeighbors),
        nameof(CoresetRatio),
        nameof(TargetEmbedDimension),
        nameof(AnomalyThreshold),
        nameof(UseManualThreshold),
        nameof(SelectedDistanceMetric),
        nameof(UseSimdDistance),
        nameof(PatchScoreParallelism),
        nameof(FeatureMapDownscale),
        nameof(UseApproximateNearestNeighbors),
        nameof(AnnClusterCount),
        nameof(AnnProbeClusters),
    ];

    public ObservableCollection<BackboneOption> AvailableBackbones { get; } = [];

    public string OkDataPath
    {
        get => _okDataPath;
        set
        {
            if (SetField(ref _okDataPath, value))
                _settings.OkDataPath = value;
        }
    }

    public string NgDataPath
    {
        get => _ngDataPath;
        set
        {
            if (SetField(ref _ngDataPath, value))
                _settings.NgDataPath = value;
        }
    }

    public string OkTrainRatio
    {
        get => _settings.OkTrainRatio.ToString("G");
        set => UpdateDoubleSetting(value, v => _settings.OkTrainRatio = v, nameof(OkTrainRatio));
    }

    public string OkTuneRatio
    {
        get => _settings.OkTuneRatio.ToString("G");
        set => UpdateDoubleSetting(value, v => _settings.OkTuneRatio = v, nameof(OkTuneRatio));
    }

    public string OkTestRatio
    {
        get => _settings.OkTestRatio.ToString("G");
        set => UpdateDoubleSetting(value, v => _settings.OkTestRatio = v, nameof(OkTestRatio));
    }

    public string NgTuneRatio
    {
        get => _settings.NgTuneRatio.ToString("G");
        set => UpdateDoubleSetting(value, v => _settings.NgTuneRatio = v, nameof(NgTuneRatio));
    }

    public bool UseRatioSplit
    {
        get => _settings.SplitMode == DatasetSplitMode.Ratio;
        set
        {
            if (!value || _settings.SplitMode == DatasetSplitMode.Ratio)
                return;

            _settings.SplitMode = DatasetSplitMode.Ratio;
            NotifySplitModeChanged();
        }
    }

    public bool UseCountSplit
    {
        get => _settings.SplitMode == DatasetSplitMode.Count;
        set
        {
            if (!value || _settings.SplitMode == DatasetSplitMode.Count)
                return;

            _settings.SplitMode = DatasetSplitMode.Count;
            NotifySplitModeChanged();
        }
    }

    public bool ShowRatioSplit => UseRatioSplit;

    public bool ShowCountSplit => UseCountSplit;

    public string OkMemoryCount
    {
        get => _settings.OkMemoryCount.ToString();
        set => UpdateIntSetting(value, v => _settings.OkMemoryCount = v, nameof(OkMemoryCount));
    }

    public string OkTuneCount
    {
        get => _settings.OkTuneCount.ToString();
        set => UpdateIntSetting(value, v => _settings.OkTuneCount = v, nameof(OkTuneCount));
    }

    public string NgTuneCount
    {
        get => _settings.NgTuneCount.ToString();
        set => UpdateIntSetting(value, v => _settings.NgTuneCount = v, nameof(NgTuneCount));
    }

    private void NotifySplitModeChanged()
    {
        OnPropertyChanged(nameof(UseRatioSplit));
        OnPropertyChanged(nameof(UseCountSplit));
        OnPropertyChanged(nameof(ShowRatioSplit));
        OnPropertyChanged(nameof(ShowCountSplit));
    }

    public string SplitSeed
    {
        get => _settings.SplitSeed.ToString();
        set => UpdateIntSetting(value, v => _settings.SplitSeed = v, nameof(SplitSeed));
    }

    public bool AutoSearchNeighbors
    {
        get => _autoSearchNeighbors;
        set
        {
            if (SetField(ref _autoSearchNeighbors, value))
                _settings.AutoSearchNeighbors = value;
        }
    }

    public string TuneResultText
    {
        get => _tuneResultText;
        set => SetField(ref _tuneResultText, value);
    }

    public string TestResultText
    {
        get => _testResultText;
        set => SetField(ref _testResultText, value);
    }

    public string ModelOutputDir
    {
        get => _modelOutputDir;
        set
        {
            if (SetField(ref _modelOutputDir, value))
            {
                _settings.ModelOutputDir = value;
                SyncModelPathsFromProfile();
            }
        }
    }

    public string ProfileOutputHint =>
        string.IsNullOrWhiteSpace(_activeProfileName)
            ? string.Empty
            : $"当前配置输出: {AppPaths.Resolve(_settings.ModelOutputDir)}/{ProfileOutputLayout.SanitizeProfileName(_activeProfileName)}/patchcore_model.json, config.json";

    public string TrainLog
    {
        get => _trainLog;
        set => SetField(ref _trainLog, value);
    }

    public string TrainTimeText
    {
        get => _trainTimeText;
        set => SetField(ref _trainTimeText, value);
    }

    public string ModelPath
    {
        get => _modelPath;
        set => SetField(ref _modelPath, value);
    }

    public string InferInputPath
    {
        get => _inferInputPath;
        set => SetField(ref _inferInputPath, value);
    }

    public string InferOutputPath
    {
        get => _inferOutputPath;
        set => SetField(ref _inferOutputPath, value);
    }

    public bool InferSingleImage
    {
        get => _inferSingleImage;
        set
        {
            if (SetField(ref _inferSingleImage, value))
                OnPropertyChanged(nameof(InferInputLabel));
        }
    }

    public bool InferDirectory
    {
        get => !_inferSingleImage;
        set => InferSingleImage = !value;
    }

    public string InferInputLabel => InferSingleImage ? "图像文件" : "图像目录";

    public string SelectedDistanceMetric
    {
        get => _settings.DistanceMetric.ToString();
        set
        {
            if (Enum.TryParse<DistanceMetric>(value, out var metric) && _settings.DistanceMetric != metric)
            {
                _settings.DistanceMetric = metric;
                OnPropertyChanged();
            }
        }
    }

    public bool UseSimdDistance
    {
        get => _settings.UseSimdDistance;
        set
        {
            if (_settings.UseSimdDistance != value)
            {
                _settings.UseSimdDistance = value;
                OnPropertyChanged();
            }
        }
    }

    public string PatchScoreParallelism
    {
        get => _settings.PatchScoreParallelism.ToString();
        set => UpdateIntSetting(value, v => _settings.PatchScoreParallelism = v, nameof(PatchScoreParallelism));
    }

    public string FeatureMapDownscale
    {
        get => _settings.FeatureMapDownscale.ToString();
        set => UpdateIntSetting(value, v => _settings.FeatureMapDownscale = v, nameof(FeatureMapDownscale));
    }

    public bool UseApproximateNearestNeighbors
    {
        get => _settings.UseApproximateNearestNeighbors;
        set
        {
            if (_settings.UseApproximateNearestNeighbors != value)
            {
                _settings.UseApproximateNearestNeighbors = value;
                OnPropertyChanged();
            }
        }
    }

    public string AnnClusterCount
    {
        get => _settings.AnnClusterCount.ToString();
        set => UpdateIntSetting(value, v => _settings.AnnClusterCount = v, nameof(AnnClusterCount));
    }

    public string AnnProbeClusters
    {
        get => _settings.AnnProbeClusters.ToString();
        set => UpdateIntSetting(value, v => _settings.AnnProbeClusters = v, nameof(AnnProbeClusters));
    }

    public bool SaveHeatmap
    {
        get => _settings.SaveHeatmap;
        set
        {
            if (_settings.SaveHeatmap != value)
            {
                _settings.SaveHeatmap = value;
                OnPropertyChanged();
            }
        }
    }

    public string InferTimeText
    {
        get => _inferTimeText;
        set => SetField(ref _inferTimeText, value);
    }

    public string InferSummary
    {
        get => _inferSummary;
        set => SetField(ref _inferSummary, value);
    }

    public string InferTimingDetail
    {
        get => _inferTimingDetail;
        set => SetField(ref _inferTimingDetail, value);
    }

    public string? PreviewOriginalPath
    {
        get => _previewOriginalPath;
        set => SetField(ref _previewOriginalPath, value);
    }

    public string? PreviewHeatmapPath
    {
        get => _previewHeatmapPath;
        set
        {
            if (SetField(ref _previewHeatmapPath, value))
                OnPropertyChanged(nameof(HasPreviewHeatmap));
        }
    }

    public bool HasPreviewHeatmap =>
        !string.IsNullOrWhiteSpace(_previewHeatmapPath) && File.Exists(_previewHeatmapPath);

    public void SetPreviewFromRow(PredictionRowViewModel row)
    {
        PreviewOriginalPath = row.ImagePath;
        PreviewHeatmapPath = row.HeatmapPath;
        InferTimingDetail = row.TimingDetail;
    }

    public void ClearPreview()
    {
        PreviewOriginalPath = null;
        PreviewHeatmapPath = null;
    }

    public bool IsBusy
    {
        get => _isBusy;
        set
        {
            if (SetField(ref _isBusy, value))
                System.Windows.Input.CommandManager.InvalidateRequerySuggested();
        }
    }

    public string ParamsConfigDir
    {
        get => _paramsConfigDir;
        set
        {
            if (!SetField(ref _paramsConfigDir, value))
                return;

            RefreshProfiles();
            LoadStartupProfile();
        }
    }

    public string ProfileName
    {
        get => _profileName;
        set => SetField(ref _profileName, value);
    }

    public string SelectedProfile
    {
        get => _selectedProfile;
        set
        {
            if (string.Equals(_selectedProfile, value, StringComparison.OrdinalIgnoreCase))
                return;

            if (_suppressProfileSwitch)
            {
                _selectedProfile = value ?? string.Empty;
                OnPropertyChanged();
                OnPropertyChanged(nameof(ActiveProfileHint));
                return;
            }

            var previous = _selectedProfile;
            if (!TrySwitchProfile(value))
            {
                _suppressProfileSwitch = true;
                _selectedProfile = previous;
                OnPropertyChanged();
                OnPropertyChanged(nameof(ActiveProfileHint));
                _suppressProfileSwitch = false;
                return;
            }

            _selectedProfile = value ?? string.Empty;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ActiveProfileHint));
        }
    }

    public string ActiveProfileHint =>
        string.IsNullOrWhiteSpace(_activeProfileName)
            ? string.Empty
            : _isProfileDirty
                ? $"当前配置: {_activeProfileName}（未保存）"
                : $"当前配置: {_activeProfileName}";

    public string SelectedBackboneId
    {
        get => _selectedBackboneId;
        set
        {
            if (!SetField(ref _selectedBackboneId, value))
                return;

            _settings.BackboneId = value;
            _settings.SyncBackbonePath();
            OnPropertyChanged(nameof(BackboneOnnxPath));
            OnPropertyChanged(nameof(IsCustomBackbone));
            UpdateBackboneUi();
            System.Windows.Input.CommandManager.InvalidateRequerySuggested();
        }
    }

    public bool IsCustomBackbone =>
        string.Equals(SelectedBackboneId, BackboneCatalog.CustomId, StringComparison.OrdinalIgnoreCase);

    public string BackboneOnnxPath
    {
        get => IsCustomBackbone ? _settings.CustomBackboneOnnxPath : _settings.BackboneOnnxPath;
        set
        {
            if (IsCustomBackbone)
            {
                _settings.CustomBackboneOnnxPath = value;
                _settings.BackboneOnnxPath = value;
            }
            else
            {
                _settings.BackboneOnnxPath = value;
            }

            OnPropertyChanged();
            UpdateBackboneUi();
        }
    }

    public string BackboneStatus
    {
        get => _backboneStatus;
        set => SetField(ref _backboneStatus, value);
    }

    public string BackboneDescription
    {
        get => _backboneDescription;
        set => SetField(ref _backboneDescription, value);
    }

    public string BackboneExportHint
    {
        get => _backboneExportHint;
        set => SetField(ref _backboneExportHint, value);
    }

    public string ImageSize
    {
        get => _settings.ImageSize.ToString();
        set => UpdateIntSetting(value, v => _settings.ImageSize = v, nameof(ImageSize));
    }

    public string PatchSize
    {
        get => _settings.PatchSize.ToString();
        set => UpdateIntSetting(value, v => _settings.PatchSize = v, nameof(PatchSize));
    }

    public string NumNeighbors
    {
        get => _settings.NumNeighbors.ToString();
        set => UpdateIntSetting(value, v => _settings.NumNeighbors = v, nameof(NumNeighbors));
    }

    public string CoresetRatio
    {
        get => _settings.CoresetRatio.ToString("G");
        set
        {
            if (_settings.CoresetRatio.ToString("G") != value)
            {
                if (double.TryParse(value, out var parsed))
                    _settings.CoresetRatio = parsed;
                OnPropertyChanged();
            }
        }
    }

    public string TargetEmbedDimension
    {
        get => _settings.TargetEmbedDimension.ToString();
        set => UpdateIntSetting(value, v => _settings.TargetEmbedDimension = v, nameof(TargetEmbedDimension));
    }

    public string AnomalyThreshold
    {
        get => _settings.AnomalyThreshold.ToString("G");
        set
        {
            if (_settings.AnomalyThreshold.ToString("G") != value)
            {
                if (float.TryParse(value, out var parsed))
                    _settings.AnomalyThreshold = parsed;
                OnPropertyChanged();
            }
        }
    }

    public bool UseManualThreshold
    {
        get => _settings.UseManualThreshold;
        set
        {
            if (_settings.UseManualThreshold != value)
            {
                _settings.UseManualThreshold = value;
                OnPropertyChanged();
            }
        }
    }

    public string ParamsStatus
    {
        get => _paramsStatus;
        set => SetField(ref _paramsStatus, value);
    }

    public ObservableCollection<PredictionRowViewModel> Predictions { get; } = [];
    public ObservableCollection<string> AvailableProfiles { get; } = [];

    public RelayCommand BrowseOkDataCommand { get; }
    public RelayCommand BrowseNgDataCommand { get; }
    public RelayCommand BrowseModelOutputCommand { get; }
    public AsyncRelayCommand TrainCommand { get; }
    public RelayCommand BrowseModelCommand { get; }
    public RelayCommand BrowseInferInputCommand { get; }
    public RelayCommand BrowseInferOutputCommand { get; }
    public AsyncRelayCommand PredictCommand { get; }

    public RelayCommand BrowseParamsConfigDirCommand { get; }
    public RelayCommand BrowseBackboneCommand { get; }
    public RelayCommand SaveParamsCommand { get; }
    public RelayCommand SaveProfileAsCommand { get; }
    public RelayCommand NewProfileCommand { get; }
    public RelayCommand DeleteProfileCommand { get; }
    public RelayCommand ResetParamsCommand { get; }
    public RelayCommand RefreshProfilesCommand { get; }

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool CanRunTrain() =>
        !IsBusy
        && Directory.Exists(OkDataPath)
        && !string.IsNullOrWhiteSpace(ModelOutputDir)
        && TryBuildConfig(out _);

    private bool CanRunPredict() =>
        !IsBusy
        && File.Exists(ModelPath)
        && !string.IsNullOrWhiteSpace(InferInputPath)
        && (InferSingleImage ? File.Exists(InferInputPath) : Directory.Exists(InferInputPath))
        && TryBuildConfig(out _);

    private void UpdateBackboneUi()
    {
        var option = BackboneCatalog.Get(SelectedBackboneId);
        BackboneDescription = option.Description;
        BackboneStatus = BackboneCatalog.GetStatusText(
            SelectedBackboneId,
            IsCustomBackbone ? _settings.CustomBackboneOnnxPath : null);
        BackboneExportHint = IsCustomBackbone
            ? "自定义模式：请浏览选择已导出的 ONNX 文件"
            : $"导出命令: {BackboneCatalog.GetExportCommand(SelectedBackboneId)}";
    }

    private void BrowseOkData()
    {
        var path = PickFolder("选择 OK 样本目录（正常产品图像）");
        if (!string.IsNullOrWhiteSpace(path))
            OkDataPath = path;
    }

    private void BrowseNgData()
    {
        var path = PickFolder("选择 NG 样本目录（异常/缺陷图像）");
        if (!string.IsNullOrWhiteSpace(path))
            NgDataPath = path;
    }

    private void BrowseModelOutput()
    {
        var path = PickFolder("选择模型输出根目录（其下按配置名分子目录）");
        if (!string.IsNullOrWhiteSpace(path))
            ModelOutputDir = path;
    }

    private void BrowseModel()
    {
        var dialog = new OpenFileDialog
        {
            Title = "选择模型 (patchcore_model.json)",
            Filter = "PatchCore 模型 (patchcore_model.json)|patchcore_model.json|JSON 文件 (*.json)|*.json",
            InitialDirectory = GetInitialDirectory(ModelPath)
        };

        if (dialog.ShowDialog() != true)
            return;

        try
        {
            var model = PatchCoreModel.Load(dialog.FileName);
            ModelPath = dialog.FileName;
            AppendTrainLog($"已加载模型: {dialog.FileName} (Memory Bank={model.MemoryBank.Length})");
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.Message, "模型无效", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void BrowseInferInput()
    {
        if (InferSingleImage)
        {
            var dialog = new OpenFileDialog
            {
                Title = "选择图像",
                Filter = "图像文件|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp",
                InitialDirectory = GetInitialDirectory(InferInputPath)
            };

            if (dialog.ShowDialog() == true)
                InferInputPath = dialog.FileName;
        }
        else
        {
            var path = PickFolder("选择推理目录");
            if (!string.IsNullOrWhiteSpace(path))
                InferInputPath = path;
        }
    }

    private void BrowseInferOutput()
    {
        var path = PickFolder("选择热力图输出目录");
        if (!string.IsNullOrWhiteSpace(path))
            InferOutputPath = path;
    }

    private void BrowseParamsConfigDir()
    {
        var path = PickFolder("选择调参配置目录");
        if (!string.IsNullOrWhiteSpace(path))
        {
            ParamsConfigDir = path;
            RefreshProfiles();
        }
    }

    private void BrowseBackbone()
    {
        var dialog = new OpenFileDialog
        {
            Title = "选择 Backbone ONNX",
            Filter = "ONNX 模型 (*.onnx)|*.onnx",
            InitialDirectory = GetInitialDirectory(BackboneOnnxPath)
        };

        if (dialog.ShowDialog() == true)
        {
            SelectedBackboneId = BackboneCatalog.CustomId;
            BackboneOnnxPath = dialog.FileName;
        }
    }

    private void SaveCurrentProfile(string? profileName = null)
    {
        var name = profileName ?? SelectedProfile;
        if (string.IsNullOrWhiteSpace(name))
        {
            MessageBox.Show("请先选择或输入配置名称。", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        if (!TryBuildConfig(out var error))
        {
            MessageBox.Show(error, "参数错误", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        try
        {
            CaptureSettingsFromUi(name);
            SettingsStore.Save(ParamsConfigDir, _settings);
            ProfileOutputLayout.SaveProfileConfig(_settings.ModelOutputDir, name, _settings);
            SettingsStore.SaveLastActiveProfile(ParamsConfigDir, name);
            _activeProfileName = name;
            _isProfileDirty = false;
            SetSelectedProfileSilently(name);
            ProfileName = name;
            RefreshProfiles();
            ParamsStatus = $"已保存: {SettingsStore.GetProfilePath(ParamsConfigDir, name)}";
            OnPropertyChanged(nameof(ActiveProfileHint));
        }
        catch (Exception ex)
        {
            ParamsStatus = $"保存失败: {ex.Message}";
            MessageBox.Show(ex.Message, "保存失败", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void SaveProfileAs()
    {
        var name = string.IsNullOrWhiteSpace(ProfileName)
            ? InputDialog.Show("另存为配置", "请输入新配置名称：", SelectedProfile)
            : ProfileName.Trim();

        if (string.IsNullOrWhiteSpace(name))
            return;

        SaveCurrentProfile(name);
    }

    private void NewProfile()
    {
        if (_isProfileDirty && !ConfirmDiscardOrSave("新建配置"))
            return;

        var name = InputDialog.Show("新建配置", "请输入配置名称：", $"profile_{AvailableProfiles.Count + 1}");
        if (string.IsNullOrWhiteSpace(name))
            return;

        var path = SettingsStore.GetProfilePath(ParamsConfigDir, name);
        if (File.Exists(path))
        {
            MessageBox.Show($"配置「{name}」已存在，请使用另存为或切换加载。", "提示",
                MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        ApplySettings(PatchCoreSettings.CreateDefault());
        _settings.ProfileName = name;
        _activeProfileName = name;
        ProfileName = name;
        SetSelectedProfileSilently(name);
        if (!AvailableProfiles.Contains(name))
            AvailableProfiles.Add(name);

        _isProfileDirty = true;
        ParamsStatus = $"新建配置「{name}」，请填写参数后保存";
        OnPropertyChanged(nameof(ActiveProfileHint));
    }

    private void DeleteProfile()
    {
        if (string.IsNullOrWhiteSpace(SelectedProfile))
            return;

        var name = SelectedProfile;
        var path = SettingsStore.GetProfilePath(ParamsConfigDir, name);
        if (!File.Exists(path))
        {
            MessageBox.Show($"配置不存在: {path}", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        var confirm = MessageBox.Show(
            $"确定删除配置「{name}」？\n{path}",
            "删除配置",
            MessageBoxButton.YesNo,
            MessageBoxImage.Warning);

        if (confirm != MessageBoxResult.Yes)
            return;

        try
        {
            SettingsStore.DeleteProfile(ParamsConfigDir, name);
            RefreshProfiles();
            ParamsStatus = $"已删除: {name}";

            if (AvailableProfiles.Count == 0)
            {
                EnsureDefaultConfig();
                RefreshProfiles();
            }

            if (AvailableProfiles.Count > 0)
            {
                var next = AvailableProfiles[0];
                TrySwitchProfile(next, force: true);
                SetSelectedProfileSilently(next);
            }
            else
            {
                ApplySettings(PatchCoreSettings.CreateDefault());
                _activeProfileName = "default";
                SetSelectedProfileSilently("default");
            }
        }
        catch (Exception ex)
        {
            ParamsStatus = $"删除失败: {ex.Message}";
            MessageBox.Show(ex.Message, "删除失败", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private bool TrySwitchProfile(string? profileName, bool force = false)
    {
        if (string.IsNullOrWhiteSpace(profileName))
            return false;

        if (!force &&
            string.Equals(profileName, _activeProfileName, StringComparison.OrdinalIgnoreCase) &&
            !_isProfileDirty)
            return true;

        if (_isProfileDirty && !force && !ConfirmDiscardOrSave($"切换到「{profileName}」"))
            return false;

        var path = SettingsStore.GetProfilePath(ParamsConfigDir, profileName);
        if (!File.Exists(path))
        {
            MessageBox.Show($"配置不存在: {path}", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }

        try
        {
            ApplySettings(SettingsStore.Load(path));
            SettingsStore.SaveLastActiveProfile(ParamsConfigDir, profileName);
            _activeProfileName = profileName;
            _isProfileDirty = false;
            ProfileName = profileName;
            ParamsStatus = $"已切换: {path}";
            SyncModelPathsFromProfile();
            OnPropertyChanged(nameof(ActiveProfileHint));
            return true;
        }
        catch (Exception ex)
        {
            ParamsStatus = $"切换失败: {ex.Message}";
            MessageBox.Show(ex.Message, "切换失败", MessageBoxButton.OK, MessageBoxImage.Error);
            return false;
        }
    }

    private bool ConfirmDiscardOrSave(string actionDescription)
    {
        var result = MessageBox.Show(
            $"配置「{_activeProfileName}」已修改，{actionDescription} 前是否保存？",
            "保存配置",
            MessageBoxButton.YesNoCancel,
            MessageBoxImage.Question);

        if (result == MessageBoxResult.Cancel)
            return false;

        if (result == MessageBoxResult.Yes)
            SaveCurrentProfile(_activeProfileName);

        return true;
    }

    private void CaptureSettingsFromUi(string profileName)
    {
        _settings.BackboneId = SelectedBackboneId;
        _settings.SyncBackbonePath();
        _settings.OkDataPath = OkDataPath;
        _settings.NgDataPath = NgDataPath;
        _settings.AutoSearchNeighbors = AutoSearchNeighbors;
        _settings.ModelOutputDir = ModelOutputDir;
        _settings.ProfileName = profileName;
        _settings.NormalizeModelOutputDir();
        _settings.NormalizeSplitRatios();
    }

    private void SyncModelPathsFromProfile()
    {
        _settings.ModelOutputDir = ModelOutputDir;
        _settings.NormalizeModelOutputDir();
        ModelOutputDir = _settings.ModelOutputDir;

        var profile = string.IsNullOrWhiteSpace(_activeProfileName) ? "default" : _activeProfileName;
        ModelPath = AppPaths.Resolve(_settings.GetModelPathForProfile(profile));
        OnPropertyChanged(nameof(ProfileOutputHint));
    }

    private void SetSelectedProfileSilently(string profileName)
    {
        _suppressProfileSwitch = true;
        _selectedProfile = profileName;
        OnPropertyChanged(nameof(SelectedProfile));
        OnPropertyChanged(nameof(ActiveProfileHint));
        _suppressProfileSwitch = false;
    }

    private void ResetParams()
    {
        ApplySettings(PatchCoreSettings.CreateDefault());
        _isProfileDirty = true;
        ParamsStatus = "已恢复默认参数（未保存）";
        OnPropertyChanged(nameof(ActiveProfileHint));
    }

    private void RefreshProfiles()
    {
        var current = SelectedProfile;
        AvailableProfiles.Clear();
        foreach (var profile in SettingsStore.ListProfiles(ParamsConfigDir))
            AvailableProfiles.Add(profile);

        if (AvailableProfiles.Count == 0)
            return;

        if (!string.IsNullOrWhiteSpace(_activeProfileName) &&
            AvailableProfiles.Contains(_activeProfileName, StringComparer.OrdinalIgnoreCase))
        {
            SetSelectedProfileSilently(_activeProfileName);
            return;
        }

        if (!string.IsNullOrWhiteSpace(current) &&
            AvailableProfiles.Contains(current, StringComparer.OrdinalIgnoreCase))
        {
            SetSelectedProfileSilently(current);
            return;
        }

        SetSelectedProfileSilently(AvailableProfiles[0]);
    }

    private void EnsureDefaultConfig()
    {
        Directory.CreateDirectory(ParamsConfigDir);
        var defaultPath = SettingsStore.GetProfilePath(ParamsConfigDir, "default");
        if (!File.Exists(defaultPath))
            SettingsStore.Save(ParamsConfigDir, PatchCoreSettings.CreateDefault());
    }

    private void LoadStartupProfile()
    {
        var lastProfile = SettingsStore.LoadLastActiveProfile(ParamsConfigDir);
        if (!string.IsNullOrWhiteSpace(lastProfile))
        {
            var path = SettingsStore.GetProfilePath(ParamsConfigDir, lastProfile);
            if (File.Exists(path))
            {
                TrySwitchProfile(lastProfile, force: true);
                SetSelectedProfileSilently(lastProfile);
                return;
            }
        }

        var defaultPath = SettingsStore.GetProfilePath(ParamsConfigDir, "default");
        if (!File.Exists(defaultPath))
            return;

        TrySwitchProfile("default", force: true);
        SetSelectedProfileSilently("default");
    }

    private void ApplySettings(PatchCoreSettings settings)
    {
        _suppressDirty = true;
        try
        {
            settings.NormalizeLegacyPaths();
            _settings.ProfileName = settings.ProfileName;
            _settings.BackboneId = string.IsNullOrWhiteSpace(settings.BackboneId)
                ? BackboneCatalog.GetByOnnxPath(settings.BackboneOnnxPath).Id
                : settings.BackboneId;
            _settings.CustomBackboneOnnxPath = settings.CustomBackboneOnnxPath;
            _settings.BackboneOnnxPath = settings.BackboneOnnxPath;
            _settings.ImageSize = settings.ImageSize;
            _settings.PatchSize = settings.PatchSize;
            _settings.NumNeighbors = settings.NumNeighbors;
            _settings.CoresetRatio = settings.CoresetRatio;
            _settings.TargetEmbedDimension = settings.TargetEmbedDimension;
            _settings.AnomalyThreshold = settings.AnomalyThreshold;
            _settings.UseManualThreshold = settings.UseManualThreshold;
            _settings.OkDataPath = settings.OkDataPath;
            _settings.NgDataPath = settings.NgDataPath;
            settings.NormalizeSplitRatios();
            _settings.OkTrainRatio = settings.OkTrainRatio;
            _settings.OkTuneRatio = settings.OkTuneRatio;
            _settings.OkTestRatio = settings.OkTestRatio;
            _settings.NgTuneRatio = settings.NgTuneRatio;
            _settings.SplitMode = settings.SplitMode;
            _settings.OkMemoryCount = settings.OkMemoryCount;
            _settings.OkTuneCount = settings.OkTuneCount;
            _settings.NgTuneCount = settings.NgTuneCount;
            _settings.SplitSeed = settings.SplitSeed;
            _settings.AutoSearchNeighbors = settings.AutoSearchNeighbors;
            _settings.UseGpu = settings.UseGpu;
            _settings.GpuDeviceId = settings.GpuDeviceId;
            _settings.InferenceBatchSize = settings.InferenceBatchSize;
            _settings.PreprocessParallelism = settings.PreprocessParallelism;
            _settings.SaveHeatmap = settings.SaveHeatmap;
            _settings.DistanceMetric = settings.DistanceMetric;
            _settings.UseSimdDistance = settings.UseSimdDistance;
            _settings.PatchScoreParallelism = settings.PatchScoreParallelism;
            _settings.FeatureMapDownscale = settings.FeatureMapDownscale;
            _settings.UseApproximateNearestNeighbors = settings.UseApproximateNearestNeighbors;
            _settings.AnnClusterCount = settings.AnnClusterCount;
            _settings.AnnProbeClusters = settings.AnnProbeClusters;
            settings.NormalizeModelOutputDir();
            _settings.ModelOutputDir = settings.ModelOutputDir;
            _settings.SyncBackbonePath();

            ProfileName = settings.ProfileName;
            _activeProfileName = settings.ProfileName;
            SelectedBackboneId = _settings.BackboneId;
            OkDataPath = settings.OkDataPath;
            NgDataPath = settings.NgDataPath;
            OnPropertyChanged(nameof(OkTrainRatio));
            OnPropertyChanged(nameof(OkTuneRatio));
            OnPropertyChanged(nameof(OkTestRatio));
            OnPropertyChanged(nameof(NgTuneRatio));
            NotifySplitModeChanged();
            OnPropertyChanged(nameof(OkMemoryCount));
            OnPropertyChanged(nameof(OkTuneCount));
            OnPropertyChanged(nameof(NgTuneCount));
            OnPropertyChanged(nameof(SplitSeed));
            AutoSearchNeighbors = settings.AutoSearchNeighbors;
            ModelOutputDir = _settings.ModelOutputDir;
            SyncModelPathsFromProfile();
            OnPropertyChanged(nameof(ImageSize));
            OnPropertyChanged(nameof(PatchSize));
            OnPropertyChanged(nameof(NumNeighbors));
            OnPropertyChanged(nameof(CoresetRatio));
            OnPropertyChanged(nameof(TargetEmbedDimension));
            OnPropertyChanged(nameof(AnomalyThreshold));
            OnPropertyChanged(nameof(UseManualThreshold));
            OnPropertyChanged(nameof(SaveHeatmap));
            OnPropertyChanged(nameof(SelectedDistanceMetric));
            OnPropertyChanged(nameof(UseSimdDistance));
            OnPropertyChanged(nameof(PatchScoreParallelism));
            OnPropertyChanged(nameof(FeatureMapDownscale));
            OnPropertyChanged(nameof(UseApproximateNearestNeighbors));
            OnPropertyChanged(nameof(AnnClusterCount));
            OnPropertyChanged(nameof(AnnProbeClusters));
            OnPropertyChanged(nameof(BackboneOnnxPath));
            UpdateBackboneUi();
            _isProfileDirty = false;
        }
        finally
        {
            _suppressDirty = false;
        }
    }

    private async Task TrainAsync()
    {
        if (!TryBuildConfig(out var error))
        {
            MessageBox.Show(error, "参数错误", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        IsBusy = true;
        TrainLog = string.Empty;
        TuneResultText = string.Empty;
        TestResultText = string.Empty;
        TrainTimeText = "耗时: 运行中...";

        try
        {
            var config = _settings.ToConfig();
            var progress = new Progress<string>(msg => AppendTrainLog(msg));
            _settings.NormalizeSplitRatios();
            CaptureSettingsFromUi(_activeProfileName);
            var result = await Task.Run(() => _service.TrainAndTune(new TrainAndTuneRequest
            {
                OkDataPath = OkDataPath,
                NgDataPath = string.IsNullOrWhiteSpace(NgDataPath) ? null : NgDataPath,
                ModelOutputDir = _settings.ModelOutputDir,
                ProfileName = _activeProfileName,
                Config = config,
                AutoSearchNeighbors = AutoSearchNeighbors,
                SplitOptions = _settings.ToSplitOptions()
            }, progress));

            TrainTimeText = $"耗时: {FormatElapsed(result.Elapsed)}";
            AppendTrainLog($"Backbone: {BackboneCatalog.Get(SelectedBackboneId).DisplayName}");
            AppendTrainLog($"输出目录: {ProfileOutputLayout.GetProfileDirectory(_settings.ModelOutputDir, _activeProfileName)}");
            AppendTrainLog($"训练完成: {result.ImageCount} 张, Memory Bank={result.MemoryBankSize}");
            AppendTrainLog($"Coreset={config.CoresetRatio:P0}, 输入={config.ImageSize}px");
            AppendTrainLog($"模型: {result.ModelPath}");
            ProfileOutputLayout.SaveProfileConfig(_settings.ModelOutputDir, _activeProfileName, _settings);
            AppendTrainLog($"配置: {_settings.GetConfigPathForProfile(_activeProfileName)}");

            if (result.TuningMetrics is { } m)
            {
                _settings.NumNeighbors = m.NumNeighbors;
                _settings.AnomalyThreshold = m.Threshold;
                OnPropertyChanged(nameof(NumNeighbors));
                OnPropertyChanged(nameof(AnomalyThreshold));

                TuneResultText =
                    $"调参完成 | 阈值={m.Threshold:F4} | kNN={m.NumNeighbors} | " +
                    $"F1={m.F1:P1} | 准确率={m.Accuracy:P1} | 精确率={m.Precision:P1} | 召回率={m.Recall:P1} | " +
                    $"TP={m.TruePositive} TN={m.TrueNegative} FP={m.FalsePositive} FN={m.FalseNegative}";
                AppendTrainLog(TuneResultText);
            }
            else
            {
                TuneResultText = $"未调参，使用训练集 P95 阈值={result.Threshold:F4}, kNN={result.NumNeighbors}";
                AppendTrainLog(TuneResultText);
            }

            if (result.TestMetrics is { } test)
            {
                TestResultText =
                    $"测试集 | F1={test.F1:P1} | 准确率={test.Accuracy:P1} | 精确率={test.Precision:P1} | 召回率={test.Recall:P1} | " +
                    $"TP={test.TruePositive} TN={test.TrueNegative} FP={test.FalsePositive} FN={test.FalseNegative}";
                AppendTrainLog(TestResultText);
            }
            else
            {
                TestResultText = string.Empty;
            }

            ModelPath = result.ModelPath;
            _isProfileDirty = true;
            OnPropertyChanged(nameof(ActiveProfileHint));
        }
        catch (Exception ex)
        {
            TrainTimeText = "耗时: 失败";
            AppendTrainLog($"错误: {ex.Message}");
            MessageBox.Show(ex.Message, "训练失败", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private async Task PredictAsync()
    {
        if (!TryBuildConfig(out var error))
        {
            MessageBox.Show(error, "参数错误", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        IsBusy = true;
        Predictions.Clear();
        ClearPreview();
        InferTimeText = "总耗时: 运行中...";
        InferSummary = string.Empty;
        InferTimingDetail = string.Empty;

        try
        {
            IEnumerable<string> inputs = InferSingleImage
                ? [InferInputPath]
                : ImagePreprocessor.EnumerateImages(InferInputPath);

            var inputList = inputs.ToList();
            if (inputList.Count == 0)
                throw new InvalidOperationException("未找到可推理的图像。");

            var config = _settings.ToConfig();
            var progress = new Progress<string>(msg => InferSummary = msg);
            var batch = await Task.Run(() =>
                _service.Predict(ModelPath, inputList, InferOutputPath, config, progress));

            foreach (var item in batch.Items)
            {
                var timing = item.Result.StageTiming;
                Predictions.Add(new PredictionRowViewModel
                {
                    FileName = Path.GetFileName(item.Result.ImagePath),
                    ImagePath = item.Result.ImagePath,
                    HeatmapPath = item.Result.HeatmapPath,
                    Score = item.Result.AnomalyScore,
                    Label = item.Result.Label,
                    ElapsedText = FormatElapsed(item.Elapsed),
                    TimingDetail = timing is null
                        ? string.Empty
                        : $"{timing.FormatStages()}{Environment.NewLine}{timing.FormatPercentages()}",
                });
            }

            var ngCount = batch.Items.Count(x => x.Result.IsAnomaly);
            InferTimeText = $"总耗时: {FormatElapsed(batch.TotalElapsed)}";
            var thresholdHint = config.UseManualThreshold
                ? $"手动阈值={config.AnomalyThreshold:F4}"
                : "模型自动阈值";
            InferSummary =
                $"Backbone={BackboneCatalog.Get(SelectedBackboneId).DisplayName} | 完成 {batch.Items.Count} 张 | NG={ngCount} | OK={batch.Items.Count - ngCount} | {thresholdHint}" +
                (config.SaveHeatmap ? "" : " | 仅结果（无热力图）");
            if (Predictions.LastOrDefault() is { } last)
                SetPreviewFromRow(last);
        }
        catch (Exception ex)
        {
            InferTimeText = "总耗时: 失败";
            InferSummary = ex.Message;
            MessageBox.Show(ex.Message, "推理失败", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool TryBuildConfig(out string error)
    {
        error = string.Empty;

        if (!int.TryParse(ImageSize, out var imageSize) || imageSize < 64 || imageSize > 1024)
        {
            error = "输入尺寸必须是 64~1024 的整数。";
            return false;
        }

        if (!int.TryParse(PatchSize, out var patchSize) || patchSize < 1 || patchSize % 2 == 0)
        {
            error = "Patch 大小必须是大于 0 的奇数。";
            return false;
        }

        if (!int.TryParse(NumNeighbors, out var neighbors) || neighbors < 1)
        {
            error = "kNN 邻居数必须是大于 0 的整数。";
            return false;
        }

        if (!double.TryParse(CoresetRatio, out var coreset) || coreset <= 0 || coreset > 1)
        {
            error = "Coreset 比例必须是 0~1 之间的小数。";
            return false;
        }

        if (!int.TryParse(TargetEmbedDimension, out var embedDim) || embedDim < 1)
        {
            error = "特征维度必须是大于 0 的整数。";
            return false;
        }

        if (!float.TryParse(AnomalyThreshold, out _))
        {
            error = "异常阈值必须是有效数字。";
            return false;
        }

        if (!Enum.TryParse<DistanceMetric>(SelectedDistanceMetric, out _))
        {
            error = "距离度量必须是 Euclidean 或 SquaredEuclidean。";
            return false;
        }

        if (!int.TryParse(PatchScoreParallelism, out var patchParallelism) || patchParallelism < 0)
        {
            error = "Patch 并行度必须是非负整数（0 表示自动）。";
            return false;
        }

        if (!int.TryParse(FeatureMapDownscale, out var downscale) || downscale < 1 || downscale > 8)
        {
            error = "特征图降采样倍数必须是 1~8 的整数。";
            return false;
        }

        if (!int.TryParse(AnnClusterCount, out var annClusters) || annClusters < 2)
        {
            error = "ANN 聚类数至少为 2。";
            return false;
        }

        if (!int.TryParse(AnnProbeClusters, out var annProbes) || annProbes < 1)
        {
            error = "ANN 探测簇数至少为 1。";
            return false;
        }

        if (_settings.SplitMode == DatasetSplitMode.Count)
        {
            if (!int.TryParse(OkMemoryCount, out var okMemoryCount) || okMemoryCount < 1)
            {
                error = "固定数量模式下 OK Memory 数量至少为 1。";
                return false;
            }

            if (!int.TryParse(OkTuneCount, out var okTuneCount) || okTuneCount < 0)
            {
                error = "OK 调参数量必须是非负整数。";
                return false;
            }

            if (!int.TryParse(NgTuneCount, out var ngTuneCount) || ngTuneCount < 0)
            {
                error = "NG 调参数量必须是非负整数。";
                return false;
            }
        }
        else
        {
            if (!double.TryParse(OkTrainRatio, out var okTrainRatio) || okTrainRatio < 0 ||
                !double.TryParse(OkTuneRatio, out var okTuneRatio) || okTuneRatio < 0 ||
                !double.TryParse(OkTestRatio, out var okTestRatio) || okTestRatio < 0)
            {
                error = "OK 划分比例必须是有效非负数。";
                return false;
            }

            if (okTrainRatio + okTuneRatio + okTestRatio <= 0)
            {
                error = "OK Memory/调参/测试比例之和必须大于 0。";
                return false;
            }

            if (!double.TryParse(NgTuneRatio, out var ngTuneRatio) || ngTuneRatio <= 0 || ngTuneRatio >= 1)
            {
                error = "NG 调参比例必须是 0~1 之间的小数（不含 0 和 1）。";
                return false;
            }
        }

        if (!int.TryParse(SplitSeed, out _))
        {
            error = "划分随机种子必须是整数。";
            return false;
        }

        if (string.IsNullOrWhiteSpace(OkDataPath) || !Directory.Exists(OkDataPath))
        {
            error = "请指定有效的 OK 样本目录。";
            return false;
        }

        _settings.BackboneId = SelectedBackboneId;
        _settings.SyncBackbonePath();
        var onnxPath = BackboneCatalog.ResolveOnnxPath(
            SelectedBackboneId,
            IsCustomBackbone ? _settings.CustomBackboneOnnxPath : null);

        if (!File.Exists(onnxPath))
        {
            error = $"Backbone ONNX 不存在: {onnxPath}\n请运行: {BackboneCatalog.GetExportCommand(SelectedBackboneId)}";
            return false;
        }

        return true;
    }

    private void UpdateDoubleSetting(string value, Action<double> setter, string propertyName)
    {
        if (double.TryParse(value, out var parsed))
            setter(parsed);
        OnPropertyChanged(propertyName);
    }

    private void UpdateIntSetting(string value, Action<int> setter, string propertyName)
    {
        if (int.TryParse(value, out var parsed))
            setter(parsed);
        OnPropertyChanged(propertyName);
    }

    private void AppendTrainLog(string message)
    {
        TrainLog = string.IsNullOrEmpty(TrainLog) ? message : $"{TrainLog}\n{message}";
    }

    private static string FormatElapsed(TimeSpan elapsed)
    {
        if (elapsed.TotalSeconds < 60)
            return $"{elapsed.TotalSeconds:F2} 秒";

        return $"{(int)elapsed.TotalMinutes} 分 {elapsed.Seconds} 秒 ({elapsed.TotalSeconds:F1} 秒)";
    }

    private static string? PickFolder(string title)
    {
        var dialog = new OpenFolderDialog { Title = title };
        return dialog.ShowDialog() == true ? dialog.FolderName : null;
    }

    private static string GetInitialDirectory(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return AppPaths.ProjectRoot;

        var resolved = AppPaths.Resolve(path);
        if (Directory.Exists(resolved))
            return resolved;

        if (File.Exists(resolved))
            return Path.GetDirectoryName(resolved) ?? AppPaths.ProjectRoot;

        var dir = Path.GetDirectoryName(resolved);
        return !string.IsNullOrWhiteSpace(dir) && Directory.Exists(dir)
            ? dir
            : AppPaths.ProjectRoot;
    }

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
            return false;

        field = value;
        OnPropertyChanged(propertyName);
        return true;
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        if (!_suppressDirty && propertyName != null && DirtyProperties.Contains(propertyName))
        {
            _isProfileDirty = true;
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ActiveProfileHint)));
        }

        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
