# Builds the HIP engine on Windows from a clean tree: ragon-run.exe, the host tools and the host tests.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows_hip.ps1 -Build <dir> [-Test]
#     [-HipPath <TheRock build or install>] [-VcpkgRoot <vcpkg>]
#
# Needs: Visual Studio 2022 (or its Build Tools) with the C++ workload, for vcvars64, cmake and ninja;
# -HipPath (default HIP_PATH) = a TheRock build or install (bin\ and lib\llvm\bin\clang++.exe); -VcpkgRoot
# (default VCPKG_ROOT) = a vcpkg with the x64-windows triplet holding icu, pcre2, nlohmann-json, libjpeg-turbo,
# libpng and libwebp. -Test runs ctest with vcpkg's DLLs on PATH. The build directory is created fresh; nothing
# outside it is written.
param(
  [Parameter(Mandatory = $true)][string]$Build,
  [string]$Source = '',
  [string]$HipPath = $env:HIP_PATH,
  [string]$VcpkgRoot = $env:VCPKG_ROOT,
  [switch]$Test,
  [switch]$KeepGoing  # build every target that can be built, past the first failure (ninja -k 0)
)
$ErrorActionPreference = 'Stop'
if (-not $Source) { $Source = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }  # the tree this script is in
if (-not $HipPath -or -not (Test-Path "$HipPath\lib\llvm\bin\clang++.exe")) {
  throw '-HipPath (or HIP_PATH) must name a TheRock build or install'
}
if (-not $VcpkgRoot -or -not (Test-Path "$VcpkgRoot\installed\x64-windows")) {
  throw '-VcpkgRoot (or VCPKG_ROOT) must name a vcpkg with the x64-windows triplet'
}
# Both are resolved before vcvars64, which sets VCPKG_ROOT to Visual Studio's bundled vcpkg.
$hip = (Resolve-Path $HipPath).Path
$vcpkg = ((Resolve-Path $VcpkgRoot).Path -replace '\\', '/') + '/installed/x64-windows'
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'no Visual Studio with the C++ x64 tools' }

# The MSVC environment (the Windows SDK, the C runtime's libraries, lld-link's search path), with HIP_PATH's own
# tools kept off PATH so CMake and Ninja are Visual Studio's.
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -and -not $_.StartsWith($hip, [StringComparison]::OrdinalIgnoreCase) }) -join ';'
$cmake = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

$h = $hip -replace '\\', '/'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& $cmake -S $Source -B $Build -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" -DRAGON_BACKEND=hip -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_CXX_COMPILER=$h/lib/llvm/bin/clang++.exe" "-DCMAKE_HIP_COMPILER=$h/lib/llvm/bin/clang++.exe" `
  "-DCMAKE_PREFIX_PATH=$vcpkg;$h" "-DCMAKE_HIP_FLAGS=--rocm-device-lib-path=$h/lib/llvm/amdgcn/bitcode" `
  "-DCMAKE_HIP_FLAGS_RELEASE=-O3 -DNDEBUG" "-DCMAKE_EXE_LINKER_FLAGS="
if ($LASTEXITCODE) { throw "configure failed ($LASTEXITCODE)" }
if ($KeepGoing) { & $cmake --build $Build -- -k 0 } else { & $cmake --build $Build }
if ($LASTEXITCODE) { throw "build failed ($LASTEXITCODE)" }
if ($Test) {
  # vcpkg's x64-windows triplet is dynamic: the executables load ICU, PCRE2 and the image codecs from its bin.
  $env:PATH = "$vcpkg/bin;$h/bin;$env:PATH" -replace '/', '\'
  & (Join-Path (Split-Path $cmake) 'ctest.exe') --test-dir $Build --output-on-failure
  if ($LASTEXITCODE) { throw "tests failed ($LASTEXITCODE)" }
}
