# Configure minitorch with MSYS2 MinGW (g++)
# Run from project root: .\configure.ps1

$ErrorActionPreference = "Stop"
$mingw = "C:\msys64\ucrt64\bin"

if (-not (Test-Path "$mingw\g++.exe")) {
    Write-Error "MSYS2 g++ not found at $mingw. Install MSYS2 and run: pacman -S mingw-w64-ucrt-x86_64-gcc"
}

if (-not (Test-Path "build")) { New-Item -ItemType Directory -Path "build" | Out-Null }
Set-Location build

# Clear cache if switching generators (e.g. NMake -> MinGW)
if (Test-Path "CMakeCache.txt") { Remove-Item CMakeCache.txt }
if (Test-Path "CMakeFiles") { Remove-Item -Recurse -Force CMakeFiles }

# Use forward slashes for CMake (backslashes cause escape errors)
$mingwPath = "C:/msys64/ucrt64/bin"
cmake -G "MinGW Makefiles" `
    -DCMAKE_CXX_COMPILER="$mingwPath/g++.exe" `
    -DCMAKE_C_COMPILER="$mingwPath/gcc.exe" `
    ..

Set-Location ..
