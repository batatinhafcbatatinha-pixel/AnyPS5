# Third-party notices

## KytyPS5 (GPL-2.0)

Parts of AnyPS5 are adapted from KytyPS5 at commit 87de64e0. KytyPS5 is licensed under the GNU
General Public License, version 2, as AnyPS5 is.

| AnyPS5 file | KytyPS5 source (commit 87de64e0) | What was adapted |
|---|---|---|
| `core/libs/prx/libSceAgcDriver/Graphics/src/BufferCache.cpp`, `Graphics/include/BufferCache.hpp` | `src/graphics/host_gpu/renderer/cache/bufferCache.cpp` (`ObtainBuffer`, `SynchronizeBuffer`, `UploadCopies`), `bufferCache.h` | Device-local guest buffer cache design: guest ranges kept in device-local buffers, upload of the changed parts only behind one barrier pair |
