#!/usr/bin/env bash
# Opens the launcher on Linux or macOS: choose your PS5 dump folder; it converts the game (AnyPS5 relinker)
# and starts it. Twin of Launcher.cmd. The launcher comes from the port-installer releases
# (source: tools/port-installer). "bash Launcher.sh --update" downloads the newest launcher again.
#   Linux: Launcher-linux-x64.AppImage     macOS: Launcher-macos.zip ("Port Launcher.app")
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
base="https://github.com/batatinhafcbatatinha-pixel/port-installer/releases/latest/download"
dir="$here/.local/launcher"
update=0
if [ "${1:-}" = "--update" ]; then update=1; fi

fail() {
  echo "$1" >&2
  echo "Get the launcher from https://github.com/batatinhafcbatatinha-pixel/port-installer/releases and put it in \"$dir/\"." >&2
  exit 1
}

fetch() { # url file
  mkdir -p "$dir"
  echo "Downloading the launcher..."
  curl -L --fail --retry 3 -sS -o "$2.part" "$1" || { rm -f "$2.part"; fail "Download failed."; }
  mv "$2.part" "$2"
}

case "$(uname -s)" in
  Linux)
    case "$(uname -m)" in
      x86_64|amd64) arch=x64 ;;
      *) fail "The launcher is published for x86-64 Linux only (this is $(uname -m))." ;;
    esac
    file="$dir/Launcher-linux-$arch.AppImage"
    if [ "$update" = 1 ]; then rm -f "$file"; fi
    if [ ! -f "$file" ]; then fetch "$base/Launcher-linux-$arch.AppImage" "$file"; fi
    chmod +x "$file"
    # Without FUSE an AppImage runs by unpacking itself first.
    if [ ! -e /dev/fuse ]; then
      export APPIMAGE_EXTRACT_AND_RUN=1
    elif ! command -v fusermount >/dev/null 2>&1 && ! command -v fusermount3 >/dev/null 2>&1; then
      export APPIMAGE_EXTRACT_AND_RUN=1
    fi
    exec "$file" --root "$here"
    ;;
  Darwin)
    app="$dir/Port Launcher.app"
    if [ "$update" = 1 ]; then rm -rf "$app"; fi
    if [ ! -d "$app" ]; then
      fetch "$base/Launcher-macos.zip" "$dir/Launcher-macos.zip"
      ditto -x -k "$dir/Launcher-macos.zip" "$dir"
      rm -f "$dir/Launcher-macos.zip"
      xattr -dr com.apple.quarantine "$app" 2>/dev/null || true
    fi
    exec open -n "$app" --args --root "$here"
    ;;
  *)
    fail "Launcher.sh supports Linux and macOS; on Windows use Launcher.cmd."
    ;;
esac
