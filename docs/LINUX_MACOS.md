# Linux and macOS (fork)

**Status: written, not run.** Nothing here was built or started on Linux or macOS by its author. CI builds the parts
that need no game (see the end). Expect to fix things on the first real run.

Upstream already supports Linux (relinker, system libraries, converted programs). This fork's additions are the
launcher path, `tools/ps5port` and the new GPU engine; they were Windows only and now have Linux and macOS paths.

## What is supported

| | Linux (x86-64) | macOS |
| --- | --- | --- |
| Launcher | `bash Launcher.sh` (downloads the port-installer AppImage) | `bash Launcher.sh` (downloads `Port Launcher.app`) |
| Convert a dump | yes: pinned AnyPS5 release v0.1.4 (relinker + `prx-linux` libraries, SHA-256 checked) | builds the relinker and libraries from this repository's sources first (`tools/ps5port/build-runtime.sh`), then converts with `--macos` |
| Play | `build/out/app.elf`, PC settings from `pc-settings.ini` as in the Windows path | `build/out/app` (x86-64, under Rosetta on Apple silicon) |
| Packages | Vulkan loader and a Vulkan driver, X11/Wayland and audio libraries (listed in `launcher.json`, per distribution) | Homebrew `cmake ninja pkg-config` and Xcode command line tools |
| GPU driver | upstream's driver (default) | upstream's driver; not known to work |

The games are never copied: `game_files` and `build/out/app0` are symbolic links to your dump.

Linux needs an x86-64 CPU with AVX2 (the converted game runs natively). On an Intel CPU `ps5port` adds `--to-intel`
to the conversion; titles that generate AMD-only instructions at run time still need an AMD CPU.

## Use

```sh
git clone --recurse-submodules https://github.com/batatinhafcbatatinha-pixel/AnyPS5.git
cd AnyPS5
bash Launcher.sh          # then File > Add game folders, Install, PLAY, as on Windows
```

`bash Launcher.sh --update` downloads the newest launcher. The launcher reads the `linux` and `macos` blocks of
`launcher.json`; it installs the listed packages (needs `pkexec` or `sudo`), runs `tools/ps5port/ps5port.py build`
and writes `Play.sh` next to `setup.json`.

Without the launcher: `python3 tools/ps5port/ps5port.py build --port <port folder>` and `... play --port <port folder>`.
The port folder needs `game_files` (your decrypted dump, with `eboot.bin` and `sce_sys`).
`ANYPS5_RELINKER` and `ANYPS5_LIBS` use your own build instead of the release, as on Windows.

## The new GPU engine

On Windows it is the default. On Linux it is **off**: the build uses the stub and the game runs on upstream's driver.
To try it, build the libraries yourself with the engine and select it at run time:

```sh
ANYPS5_EXTRA_CMAKE_ARGS="-DANYPS5_KYTY_GPU_LINUX=ON" tools/ps5port/build-runtime.sh
ANYPS5_RELINKER=.../relinker ANYPS5_LIBS=.../libs APS5_GPU_BACKEND=vram python3 tools/ps5port/ps5port.py play --port <port>
```

(the release v0.1.4 libraries do not contain the engine on any system). Linux changes for it: page protection with
`mprotect` and `/proc/self/maps` instead of `VirtualProtect`/`VirtualQuery`; page faults come from a `SIGSEGV` hook in
the libc crash handler (`LibcSetGpuFaultHandler_nid_no_patch`); Vulkan through `libvulkan.so.1`; a pthread key instead
of the Windows TLS slot; `std::shared_mutex` instead of the SRW lock; a condition variable instead of the Windows
event in the command recorder. It is the least certain part of this work. On macOS it stays a stub.

## Known not to work, or not checked

- Everything above is untested. The engine on Linux is the likeliest to fail (first compile, then run).
- macOS: there is no macOS release of the libraries; the first conversion builds them from source (hours, several GB,
  needs the git submodules). The full macOS build and a converted game have never been run by this fork. Rosetta needs
  `--to-rosetta` (set by `ps5port` on Apple silicon). The new GPU engine, patches that write memory (`"offset"`) and
  held writes do not exist on macOS.
- Linux ARM: not supported (x86-64 code).
- "Held" patch writes (`"hold": true`) use polling only on Linux (no debug-register watchpoints); a value the game
  uses the moment it writes it may be seen once with the old value.
- Patch writes into the executable's image work on the Linux output (ELF `PT_LOAD` data, address `0x400000` + offset).
- Settings that need an AnyPS5 runtime newer than v0.1.4 (frame limit, upscaling and others the launcher marks) are
  ignored by the release, as on Windows.
- The launcher's "Automatic" console language reads `LC_ALL`, `LC_MESSAGES`, `LANGUAGE` or `LANG`.
- `launcher.json` lists `apt` and `apt-get` package keys (the reference recompiler uses `apt`, the setup script of
  the port-installer looks up `apt-get`).

## CI

`.github/workflows/linux-macos.yml` (no game content): shell and JSON checks and the `ps5port` tests; the relinker
and its tests on `ubuntu-latest`, `macos-15` and `macos-15-intel`; the system libraries on Linux, with and without the
new engine (the engine job may fail without failing the workflow); the macOS libraries only when started by hand.
