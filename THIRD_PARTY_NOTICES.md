# Third-party notices

## KytyPS5 (GPL-2.0)

Parts of AnyPS5 are adapted from KytyPS5 at commit 87de64e0. KytyPS5 is licensed under the GNU
General Public License, version 2, as AnyPS5 is.

| AnyPS5 file | KytyPS5 source (commit 87de64e0) | What was adapted |
|---|---|---|
| `core/libs/prx/libSceAgcDriver/Graphics/src/BufferCache.cpp`, `Graphics/include/BufferCache.hpp` | `src/graphics/host_gpu/renderer/cache/bufferCache.cpp` (`ObtainBuffer`, `SynchronizeBuffer`, `UploadCopies`), `bufferCache.h` | Device-local guest buffer cache design: guest ranges kept in device-local buffers, upload of the changed parts only behind one barrier pair |

## KytyPS5 GPU stack (GPL-2.0; parts MIT and GPL-2.0-or-later)

The VRAM-resident GPU backend of libSceAgcDriver (`core/libs/prx/libSceAgcDriver/Kyty/`) contains
KytyPS5's guest GPU command processor, renderer and shader recompiler, copied from KytyPS5 at commit
3a7d8030ab8317781d63babcc51fa1b4d773a1a1 (https://github.com/batatinhafcbatatinha-pixel/KytyPS5, a fork of
https://github.com/KytyPS5/KytyPS5). KytyPS5 is licensed under the GNU General Public License,
version 2; its original parts are also available under the MIT license (Copyright (c) 2021
InoriRus); some recompiler and shader files derive from the shadPS4 Emulator Project
(GPL-2.0-or-later, see their SPDX headers). All of them are distributed here under GPL-2.0.

| AnyPS5 path | KytyPS5 source (commit 3a7d8030) | Changes |
|---|---|---|
| `Kyty/src/graphics/guest_gpu/**` | `src/graphics/guest_gpu/**` | AnyPS5 flip and rendering-wait packets, EOP flip tokens (`pm4Handlers.cpp`, `pm4Dispatch.*`, `graphicsRun.cpp`, `commandProcessor.h`) |
| `Kyty/src/graphics/host_gpu/**` | `src/graphics/host_gpu/**` | Interrupts delivered through libSceAgcDriver (`renderContext.cpp`); address window folding (`regionDefinitions.h`, `pageManager.cpp`, `memoryTracker.*`, `bufferCache.cpp`, `textureCache.h`); no DLSS/FSR (`upscaler/`); SPIRV-Tools optional (`pipelineCache.cpp`) |
| `Kyty/src/graphics/shader/**` | `src/graphics/shader/**` | none |
| `Kyty/src/common/**` | `src/common/**` | reduced `emulatorConfig.h`; own `logging/log.cpp`, `assert.cpp`, `file.cpp` |
| `Kyty/embed_spirv.cmake` | `src/embed_spirv.cmake` | none |
| `Kyty/glue/vulkanDevice.cpp` | `src/graphics/presentation/window/vulkanWindow.cpp` | device selection and creation without a window, no headset/DLSS/FSR |
| `Kyty/glue/Presenter.cpp` | `src/graphics/presentation/window/swapchain.cpp`, `videoOut.cpp` | frame copy and swapchain blit only |
| `Kyty/glue/Memory.cpp`, `Kyty/glue/Core.cpp` | `src/kernel/memory.cpp`, `src/loader/runtimeLinker.cpp` | SRT-walker fault path, read-back helpers |

The third-party libraries this backend needs are submodules: `3rdparty/fmt` (MIT), `3rdparty/xxHash`
(BSD-2-Clause), `3rdparty/magic_enum` (MIT), `3rdparty/xbyak` (BSD-3-Clause), at the commits KytyPS5 uses.
