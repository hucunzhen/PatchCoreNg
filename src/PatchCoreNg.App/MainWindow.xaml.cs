using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;

namespace PatchCoreNg.App;

public partial class MainWindow : Window
{
    private readonly MainViewModel _viewModel = new();

    public MainWindow()
    {
        InitializeComponent();
        DataContext = _viewModel;
        _viewModel.PropertyChanged += ViewModel_OnPropertyChanged;
    }

    private void ViewModel_OnPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(MainViewModel.TrainLog))
            ScrollTrainLogToEnd();
    }

    private void ScrollTrainLogToEnd()
    {
        if (TrainLogScrollViewer is null)
            return;

        Dispatcher.BeginInvoke(TrainLogScrollViewer.ScrollToEnd, System.Windows.Threading.DispatcherPriority.Loaded);
    }

    private void PredictionsGrid_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (sender is not DataGrid grid || grid.SelectedItem is not PredictionRowViewModel row)
            return;

        _viewModel.PreviewImagePath = row.PreviewPath;
    }
}
