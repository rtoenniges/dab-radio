<#
.SYNOPSIS
Builds native\rtlsdr.dll: librtlsdr with libusb linked in and RAW_IO switched on.

.DESCRIPTION
Why the radio needs a build of its own: without the pipe policy RAW_IO, WinUSB passes
on one read at a time however many transfers are submitted. Between two of them the
stick is not polled, its buffer overflows, and samples are lost without any notice
(measured 02.10.2026 in the GNSS receiver, see src\sdr\library.py). The librtlsdr of the
package pyrtlsdrlib is rtl-sdr 0.8.0 with a libusb that cannot set RAW_IO.

The script and rtlsdr-raw-io.patch are those of ..\gnss-receiver\native. The radio adds
rtlsdr-r82xx-if-agc.patch: rtl-sdr 2.0.3 sets the IF amplifier (VGA) of the R820T/2 to
a fixed 16.3 dB, with which no tuner gain suits both strong and weak ensembles (measured
02.10.2026, see DEFAULT_GAIN_DB in src\ui\main_window.py). The patch hands the VGA to the
IF AGC of the RTL2832U, as the library of pyrtlsdrlib does.

Sources, pinned by tag and commit:
  libusb 1.0.30, the first release whose API sets RAW_IO (libusb_endpoint_set_raw_io)
  rtl-sdr 2.0.3 of osmocom, plus the two patches

Both are compiled into one DLL with the static C runtime, so that it depends on nothing
but Windows. The sources and objects go to a folder outside OneDrive.

Needs git and Visual Studio with the C++ tools (found through vswhere).

.EXAMPLE
pwsh native\build_rtlsdr.ps1
#>
param(
    [string]$BuildDir = (Join-Path $env:LOCALAPPDATA "dab-radio\build\rtlsdr")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Libusb = @{
    Name   = "libusb"
    Url    = "https://github.com/libusb/libusb.git"
    Tag    = "v1.0.30"
    Commit = "87a55632db62c9bdc58cd31d3ccfa673f1bb017f"
}
$RtlSdr = @{
    Name   = "librtlsdr"
    Url    = "https://github.com/steve-m/librtlsdr.git"  # the maintainer's mirror of osmocom
    Tag    = "v2.0.3"
    Commit = "797f8143266d983c56d8f35d2d442527529dd8a5"
}
# Applied in this order; the second expects the line numbers the first leaves behind.
$Patches = @("rtlsdr-raw-io.patch", "rtlsdr-r82xx-if-agc.patch") |
    ForEach-Object { Join-Path $PSScriptRoot $_ }
$Output = Join-Path $PSScriptRoot "rtlsdr.dll"

function Invoke-Tool {
    # Runs a program and stops the script if it fails.
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

function Get-Source {
    # A checkout of exactly the pinned commit, without local changes. Line endings as in
    # the repository, so that the patch applies.
    param($Source)
    $dir = Join-Path $BuildDir $Source.Name
    if (-not (Test-Path (Join-Path $dir ".git"))) {
        Invoke-Tool git @("-c", "core.autocrlf=false", "-c", "advice.detachedHead=false",
            "clone", "--quiet", "--depth", "1", "--branch", $Source.Tag, $Source.Url, $dir)
    }
    $head = (& git -C $dir rev-parse HEAD).Trim()
    if ($head -ne $Source.Commit) {
        throw "$($Source.Name): $($Source.Tag) is commit $head, expected $($Source.Commit)"
    }
    Invoke-Tool git @("-C", $dir, "-c", "core.autocrlf=false", "checkout", "--quiet", "--force", "HEAD")
    return $dir
}

function Import-VisualStudio {
    # Puts the 64-bit compiler and linker of the newest Visual Studio on the path.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Visual Studio not found (no vswhere.exe)"
    }
    $root = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $root) {
        throw "No Visual Studio with the C++ tools found"
    }
    $vcvars = Join-Path $root "VC\Auxiliary\Build\vcvars64.bat"
    $lines = & cmd /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $lines) {
        if ($line -match "^([^=]+)=(.*)$") {
            Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
        }
    }
    Write-Host "Compiler: $((Get-Command cl.exe).Source)"
}

New-Item -ItemType Directory -Force $BuildDir | Out-Null
$usbDir = Get-Source $Libusb
$rtlDir = Get-Source $RtlSdr
foreach ($patch in $Patches) {
    Invoke-Tool git @("-C", $rtlDir, "apply", $patch)
}
Import-VisualStudio

$usbObjects = Join-Path $BuildDir "obj\libusb"
$rtlObjects = Join-Path $BuildDir "obj\librtlsdr"
New-Item -ItemType Directory -Force $usbObjects, $rtlObjects | Out-Null

# The files of msvc\libusb_static.vcxproj, with its definitions.
$usbSources = @(
    "core.c", "descriptor.c", "hotplug.c", "io.c", "strerror.c", "sync.c",
    "os\events_windows.c", "os\threads_windows.c", "os\windows_common.c",
    "os\windows_usbdk.c", "os\windows_winusb.c"
) | ForEach-Object { Join-Path $usbDir "libusb\$_" }
# The library target of src\CMakeLists.txt; `rtlsdr_EXPORTS` exports its API.
$rtlSources = @(
    "librtlsdr.c", "tuner_e4k.c", "tuner_fc0012.c", "tuner_fc0013.c", "tuner_fc2580.c",
    "tuner_r82xx.c"
) | ForEach-Object { Join-Path $rtlDir "src\$_" }
$common = @("/nologo", "/c", "/O2", "/MT", "/W3", "/DNDEBUG", "/D_CRT_SECURE_NO_WARNINGS")

Invoke-Tool cl (@($common) + @(
        "/utf-8", "/D_WIN32_WINNT=_WIN32_WINNT_VISTA",
        "/I$(Join-Path $usbDir 'msvc')", "/I$(Join-Path $usbDir 'libusb')",
        "/Fo$usbObjects\"
    ) + $usbSources)
# C5287 (operands of different enum types) and C4244 (double to int) are warnings of
# rtl-sdr 2.0.3 itself with this compiler.
Invoke-Tool cl (@($common) + @(
        "/wd5287", "/wd4244", "/Drtlsdr_EXPORTS",
        "/I$(Join-Path $rtlDir 'include')", "/I$(Join-Path $usbDir 'libusb')",
        "/Fo$rtlObjects\"
    ) + $rtlSources)
$dll = Join-Path $BuildDir "rtlsdr.dll"
Invoke-Tool link @("/nologo", "/DLL", "/OUT:$dll", "$usbObjects\*.obj", "$rtlObjects\*.obj")

Copy-Item $dll $Output -Force
Write-Host "Built $Output ($((Get-Item $Output).Length) bytes): rtl-sdr $($RtlSdr.Tag) with RAW_IO and IF AGC, libusb $($Libusb.Tag)"
