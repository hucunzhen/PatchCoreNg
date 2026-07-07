param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

function Resolve-CMake {
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "${env:ProgramFiles}\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe"
    )

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($installPath) {
            $candidates += Join-Path $installPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        }
    }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw @"
未找到 cmake。请安装以下任一工具后重试：
  1. CMake: https://cmake.org/download/
  2. Visual Studio（勾选「使用 C++ 的桌面开发」，含 CMake 组件）
"@
}

$cmake = Resolve-CMake
$root = Split-Path -Parent $PSScriptRoot
$native = Join-Path $root "native"
$build = Join-Path $native "build"

Write-Host "Using cmake: $cmake"

$cacheFile = Join-Path $build "CMakeCache.txt"
if (-not (Test-Path $cacheFile)) {
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        & $cmake -S $native -B $build -A x64
    } else {
        & $cmake -S $native -B $build -DCMAKE_BUILD_TYPE=$Configuration
    }
}

& $cmake --build $build --config $Configuration

$dll = Join-Path $build "bin\patchcore_native.dll"
if (-not (Test-Path $dll)) {
    $dll = Get-ChildItem -Path $build -Recurse -Filter patchcore_native.dll -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}

if (Test-Path $dll) {
    Write-Host "Built: $dll"
} else {
    throw "Native build failed: patchcore_native.dll not found under $build"
}
