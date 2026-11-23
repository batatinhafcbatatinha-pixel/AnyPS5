#!/usr/bin/env bash
# Builds the AnyPS5 relinker and the AnyPS5 system libraries from this repository's sources.
# tools/ps5port uses it on macOS (there is no macOS release); on Linux it is for people who want
# a build of their own instead of the pinned release.
#
#   tools/ps5port/build-runtime.sh [build-dir]      default: <repo>/.local/tools/anyps5-source
#   JOBS=4 tools/ps5port/build-runtime.sh           parallel jobs (default: all CPUs)
#   ANYPS5_EXTRA_CMAKE_ARGS="-DANYPS5_KYTY_GPU_LINUX=ON" tools/ps5port/build-runtime.sh   extra CMake options
#
# Result:  <build-dir>/relinker/core/relinker/relinker      the relinker
#          <build-dir>/full/core/libs/libs/                  the system libraries (libs/ of a converted game)
# Needs: git, CMake 3.24+, Ninja, a C++20 compiler (Linux: gcc/g++; macOS: Xcode command line tools),
#        and the packages of docs/dev/BUILD.md (Linux: Vulkan, X11/Wayland, audio development files).
#        macOS: brew install cmake ninja pkg-config
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
build="${1:-$repo/.local/tools/anyps5-source}"
case "$(uname -s)" in
  Darwin) cpus="$(sysctl -n hw.ncpu 2>/dev/null || echo 2)" ;;
  *) cpus="$(nproc 2>/dev/null || echo 2)" ;;
esac
jobs="${JOBS:-$cpus}"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "build-runtime: '$1' is missing. $2" >&2; exit 1; }
}
need git "Install git."
need cmake "Linux: install cmake (3.24 or newer). macOS: brew install cmake."
need ninja "Linux: install ninja-build. macOS: brew install ninja."

# The third-party sources are git submodules; a ZIP download of the repository has none of them.
if [ ! -f "$repo/3rdparty/SDL2/CMakeLists.txt" ]; then
  if [ -e "$repo/.git" ]; then
    echo "build-runtime: fetching the submodules ..."
    git -C "$repo" submodule update --init --recursive
  else
    echo "build-runtime: 3rdparty/ is empty and this folder is not a git checkout." >&2
    echo "Clone the repository with: git clone --recurse-submodules <url>" >&2
    exit 1
  fi
fi

mkdir -p "$build"
# shellcheck disable=SC2086
extra=(${ANYPS5_EXTRA_CMAKE_ARGS:-})

echo "build-runtime: relinker ..."
cmake -S "$repo" -B "$build/relinker" -G Ninja -DCMAKE_BUILD_TYPE=Release -DANYPS5_RELINKER_ONLY=ON
cmake --build "$build/relinker" --parallel "$jobs"

echo "build-runtime: system libraries (long) ..."
cmake -S "$repo" -B "$build/full" -G Ninja -DCMAKE_BUILD_TYPE=Release ${extra[@]+"${extra[@]}"}
cmake --build "$build/full" --target libs --parallel "$jobs"

test -x "$build/relinker/core/relinker/relinker"
test -f "$build/full/core/libs/libs/libkernel.prx"
echo "build-runtime: done."
echo "  relinker: $build/relinker/core/relinker/relinker"
echo "  libs:     $build/full/core/libs/libs"
