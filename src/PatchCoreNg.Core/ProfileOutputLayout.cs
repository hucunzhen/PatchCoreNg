namespace PatchCoreNg;

public static class ProfileOutputLayout
{
    public const string ModelFileName = "patchcore_model.json";
    public const string ConfigFileName = "config.json";

    public static string SanitizeProfileName(string profileName)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = profileName.Trim().Where(ch => !invalid.Contains(ch)).ToArray();
        var sanitized = new string(chars);
        return string.IsNullOrWhiteSpace(sanitized) ? "default" : sanitized;
    }

    public static string GetProfileDirectory(string outputBaseDir, string profileName)
    {
        var baseDir = AppPaths.Resolve(NormalizeOutputBaseDir(outputBaseDir));
        return Path.Combine(baseDir, SanitizeProfileName(profileName));
    }

    public static string GetModelPath(string outputBaseDir, string profileName) =>
        Path.Combine(GetProfileDirectory(outputBaseDir, profileName), ModelFileName);

    public static string GetConfigPath(string outputBaseDir, string profileName) =>
        Path.Combine(GetProfileDirectory(outputBaseDir, profileName), ConfigFileName);

    public static string EnsureProfileDirectory(string outputBaseDir, string profileName)
    {
        var dir = GetProfileDirectory(outputBaseDir, profileName);
        Directory.CreateDirectory(dir);
        return dir;
    }

    public static void SaveProfileConfig(string outputBaseDir, string profileName, PatchCoreSettings settings)
    {
        EnsureProfileDirectory(outputBaseDir, profileName);
        var path = GetConfigPath(outputBaseDir, profileName);
        settings.ProfileName = SanitizeProfileName(profileName);
        SettingsStore.SaveToFile(path, settings);
    }

    public static string NormalizeOutputBaseDir(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return "models";

        var resolved = AppPaths.Resolve(path);
        if (Directory.Exists(resolved))
            return path;

        if (File.Exists(resolved) &&
            string.Equals(Path.GetExtension(resolved), ".json", StringComparison.OrdinalIgnoreCase))
        {
            var parent = Path.GetDirectoryName(resolved);
            return string.IsNullOrWhiteSpace(parent) ? "models" : parent;
        }

        return path;
    }
}
