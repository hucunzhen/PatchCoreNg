using System.Configuration;
using System.Data;
using System.Windows;

namespace PatchCoreNg.App;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        ConsoleLog.Attach("PatchCore-NG Log");
        ConsoleLog.WriteLine(NativeAcceleration.DescribeStatus());
        base.OnStartup(e);
    }
}

