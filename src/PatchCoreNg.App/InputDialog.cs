using System.Windows;

namespace PatchCoreNg.App;

internal static class InputDialog
{
    public static string? Show(string title, string prompt, string defaultValue = "")
    {
        var input = new System.Windows.Controls.TextBox
        {
            Text = defaultValue,
            Margin = new Thickness(0, 8, 0, 0),
            MinWidth = 280
        };

        var panel = new System.Windows.Controls.StackPanel { Margin = new Thickness(16) };
        panel.Children.Add(new System.Windows.Controls.TextBlock { Text = prompt, TextWrapping = TextWrapping.Wrap });
        panel.Children.Add(input);

        var dialog = new Window
        {
            Title = title,
            Content = panel,
            SizeToContent = SizeToContent.WidthAndHeight,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            ResizeMode = ResizeMode.NoResize,
            ShowInTaskbar = false
        };

        var buttons = new System.Windows.Controls.StackPanel
        {
            Orientation = System.Windows.Controls.Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(16, 0, 16, 16)
        };

        var ok = new System.Windows.Controls.Button { Content = "确定", Width = 80, Margin = new Thickness(0, 0, 8, 0), IsDefault = true };
        var cancel = new System.Windows.Controls.Button { Content = "取消", Width = 80, IsCancel = true };
        ok.Click += (_, _) => { dialog.DialogResult = true; dialog.Close(); };
        cancel.Click += (_, _) => { dialog.DialogResult = false; dialog.Close(); };
        buttons.Children.Add(ok);
        buttons.Children.Add(cancel);

        var root = new System.Windows.Controls.StackPanel();
        root.Children.Add(panel);
        root.Children.Add(buttons);
        dialog.Content = root;

        if (Application.Current?.MainWindow is { IsLoaded: true } owner)
            dialog.Owner = owner;

        input.SelectAll();
        input.Focus();

        return dialog.ShowDialog() == true ? input.Text.Trim() : null;
    }
}
