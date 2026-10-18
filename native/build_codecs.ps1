<#
.SYNOPSIS
Builds native\faad.dll and native\mpg123.dll: the audio decoders FAAD2 and libmpg123,
from their original sources.

.DESCRIPTION
The radio decodes the audio of DAB+ with FAAD2 (AAC) and that of classic DAB with
libmpg123 (MPEG Audio Layer II), both loaded through ctypes (src\audio\aac.py,
src\audio\mp2.py). On Linux the distribution provides them (apt install libfaad2
libmpg123-0); on Windows nobody does, so this script builds the two DLLs from the
sources of the projects, unchanged, with CMake and Visual Studio. The radio looks for
them in native\ (src\audio\native.py).

Sources, pinned:
  FAAD2 2.11.4 from the project's git repository, by tag and commit. DAB+ frames of
    960 samples need nothing special: ALLOW_SMALL_FRAMELENGTH is on by default.
  mpg123 1.33.7 as the release tarball of mpg123.de, by SHA-256. Development happens in
    Subversion, and the git mirror on GitHub carries no tags; the tarball is what the
    project publishes.

Both are built with the static C runtime, so that they depend on nothing but Windows.
libmpg123 gets its generic C decoder: the assembler optimizations need yasm under MSVC,
and the few hundred kbit/s of a DAB ensemble do not need them. Only the two libraries
are built, no command-line tools. The sources and objects go to a folder outside the
repository.

Needs git and Visual Studio with the C++ tools and its CMake component ("C++ CMake
tools for Windows", which brings cmake and ninja; found through vswhere).

.EXAMPLE
pwsh native\build_codecs.ps1
#>
param(
    [string]$BuildDir = (Join-Path $env:LOCALAPPDATA "dab-radio\build\codecs")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Faad2 = @{
    Name   = "faad2"
    Url    = "https://github.com/knik0/faad2.git"
    Tag    = "2.11.4"
    Commit = "e8e76f0a44db45aed773a3f62fe35a63c867a738"
}
$Mpg123 = @{
    Name    = "mpg123-1.33.7"
    Version = "1.33.7"
    Url     = "https://www.mpg123.de/download/mpg123-1.33.7.tar.bz2"
    Sha256  = "31d0e35a4ca567ec9b5ebda6c3062bb4435d6d3eacd6ef0d95cadd7854dc03ee"
}
$Outputs = @{
    Faad   = Join-Path $PSScriptRoot "faad.dll"
    Mpg123 = Join-Path $PSScriptRoot "mpg123.dll"
}

function Invoke-Tool {
    # Runs a program, shows its output and stops the script if it fails. The output
    # goes to the host so that it does not become the return value of the caller.
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

function Get-GitSource {
    # A checkout of exactly the pinned commit, without local changes.
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

function Get-TarballSource {
    # The unpacked release tarball, downloaded once and checked against its SHA-256.
    param($Source)
    $downloads = Join-Path $BuildDir "download"
    New-Item -ItemType Directory -Force $downloads | Out-Null
    $tarball = Join-Path $downloads (Split-Path -Leaf $Source.Url)
    if (-not (Test-Path $tarball)) {
        Write-Host "Downloading $($Source.Url)"
        Invoke-WebRequest -Uri $Source.Url -OutFile $tarball
    }
    $hash = (Get-FileHash -Algorithm SHA256 $tarball).Hash.ToLowerInvariant()
    if ($hash -ne $Source.Sha256) {
        throw "$($Source.Name): $tarball has SHA-256 $hash, expected $($Source.Sha256)"
    }
    $dir = Join-Path $BuildDir $Source.Name
    if (-not (Test-Path $dir)) {
        # The tar of Windows (bsdtar) unpacks bzip2; the GNU tar of Git for Windows may
        # not be on the PATH.
        $tar = Join-Path $env:SystemRoot "System32\tar.exe"
        Invoke-Tool $tar @("-xjf", $tarball, "-C", $BuildDir)
        if (-not (Test-Path $dir)) {
            throw "$($Source.Name): the tarball did not unpack to $dir"
        }
    }
    return $dir
}

function Import-VisualStudio {
    # Puts the 64-bit compiler and linker of the newest Visual Studio on the path, with
    # the cmake and ninja of its CMake component.
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
    foreach ($tool in @("cl.exe", "cmake.exe", "ninja.exe", "dumpbin.exe")) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "$tool not found; Visual Studio needs the C++ tools and the component 'C++ CMake tools for Windows'"
        }
    }
    Write-Host "Compiler: $((Get-Command cl.exe).Source)"
    Write-Host "CMake: $((cmake --version | Select-Object -First 1) -replace 'cmake version ', '')"
}

function Build-Library {
    # Configures and builds one target of a CMake project as a DLL with the static C
    # runtime (/MT; CMP0091 lets CMAKE_MSVC_RUNTIME_LIBRARY decide also for projects
    # that require an older CMake) and returns the path of the DLL.
    param([string]$SourceDir, [string]$ObjectDir, [string]$Target, [string]$Dll, [string[]]$Options)
    Invoke-Tool cmake (@(
            "-S", $SourceDir, "-B", $ObjectDir, "-G", "Ninja", "--no-warn-unused-cli", "-Wno-dev",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
            "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW"
        ) + $Options)
    Invoke-Tool cmake @("--build", $ObjectDir, "--target", $Target)
    $built = @(Get-ChildItem -Path $ObjectDir -Recurse -Filter $Dll)
    if ($built.Count -ne 1) {
        throw "Expected one $Dll under $ObjectDir, found $($built.Count)"
    }
    return $built[0].FullName
}

function Assert-Standalone {
    # Fails if the DLL imports a C runtime DLL, i.e. was not linked with /MT.
    param([string]$Dll)
    $imports = & dumpbin /nologo /dependents $Dll |
        Where-Object { $_ -match "^\s+\S+\.dll\s*$" } | ForEach-Object { $_.Trim() }
    $runtime = $imports | Where-Object { $_ -match "^(vcruntime|msvcp|msvcr|ucrtbase|api-ms-win-crt)" }
    if ($runtime) {
        throw "$Dll depends on the C runtime: $($runtime -join ', ')"
    }
    Write-Host "  imports: $($imports -join ', ')"
}

New-Item -ItemType Directory -Force $BuildDir | Out-Null
$faadDir = Get-GitSource $Faad2
$mpg123Dir = Get-TarballSource $Mpg123
Import-VisualStudio

# FAAD2's CMake builds four variants of the library (plain, DRM, fixed point, both);
# the radio needs the plain floating-point one. No command-line tool. CMake appends the
# ABI version to the name (faad-2.dll); the radio looks for faad.dll.
$faadDll = Build-Library -SourceDir $faadDir -ObjectDir (Join-Path $BuildDir "obj\faad2") `
    -Target faad -Dll "faad-2.dll" -Options @("-DFAAD_BUILD_CLI=OFF")
# mpg123's CMake port lives in ports\cmake. Only libmpg123: neither the output library
# libout123 nor the programs, which would need it.
$mpg123Dll = Build-Library -SourceDir (Join-Path $mpg123Dir "ports\cmake") `
    -ObjectDir (Join-Path $BuildDir "obj\mpg123") -Target libmpg123 -Dll "mpg123.dll" `
    -Options @("-DBUILD_LIBOUT123=OFF", "-DBUILD_PROGRAMS=OFF")

foreach ($pair in @(@($faadDll, $Outputs.Faad), @($mpg123Dll, $Outputs.Mpg123))) {
    $dll, $output = $pair
    Assert-Standalone $dll
    Copy-Item $dll $output -Force
}
Write-Host "Built $($Outputs.Faad) ($((Get-Item $Outputs.Faad).Length) bytes): FAAD2 $($Faad2.Tag)"
Write-Host "Built $($Outputs.Mpg123) ($((Get-Item $Outputs.Mpg123).Length) bytes): libmpg123 $($Mpg123.Version)"
