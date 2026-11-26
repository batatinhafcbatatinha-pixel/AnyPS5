# The VRAM-resident GPU backend of libSceAgcDriver: KytyPS5's guest GPU command processor,
# renderer (buffer/texture caches, page tracker, pipeline cache) and shader recompiler, plus the
# glue to AnyPS5 (glue/). Included from libSceAgcDriver's CMakeLists.txt after the target exists.

# Windows, and Linux with ANYPS5_KYTY_GPU_LINUX=ON (experimental). Elsewhere (macOS, and Linux by default)
# the stub in glue_stub/ reports the backend as unavailable.
if(NOT WIN32 AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ANYPS5_KYTY_GPU_LINUX))
    target_sources(libSceAgcDriver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Kyty/glue_stub/BackendStub.cpp)
    return()
endif()

set(agcKytyRoot ${CMAKE_CURRENT_SOURCE_DIR}/Kyty)
set(agcKytySource ${agcKytyRoot}/src)
set(agcKytyShaders ${agcKytySource}/graphics/host_gpu/shaders)
set(agcKytyGenerated ${CMAKE_CURRENT_BINARY_DIR}/kyty-generated)

file(GLOB_RECURSE agcKytySources CONFIGURE_DEPENDS
        ${agcKytySource}/*.cpp
        ${agcKytyRoot}/glue/*.cpp
)

set(agcKytyTilerShaders standard256 standard4 standard4_3d standard64 standard64_3d prt prt_3d render_target depth promote_d16 demote_d16 color_transform color_clear)
file(GLOB agcKytyTilerIncludes CONFIGURE_DEPENDS "${agcKytyShaders}/gpu_tiler_*.inc")
set(agcKytyShaderHeaders "")
foreach(shaderName IN LISTS agcKytyTilerShaders)
    set(shaderSource "${agcKytyShaders}/gpu_tiler_${shaderName}.comp")
    set(shaderSpv "${agcKytyGenerated}/gpu_tiler_shaders/gpu_tiler_${shaderName}.spv")
    set(shaderHeader "${agcKytyGenerated}/gpu_tiler_shaders/gpu_tiler_${shaderName}_spv.h")
    string(TOUPPER "GPU_TILER_${shaderName}_SPV" shaderSymbol)
    add_custom_command(
            OUTPUT "${shaderHeader}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${agcKytyGenerated}/gpu_tiler_shaders"
            COMMAND "$<TARGET_FILE:glslang-standalone>" -V --target-env vulkan1.3 "-I${agcKytyShaders}" -o "${shaderSpv}" "${shaderSource}"
            COMMAND ${CMAKE_COMMAND} -DINPUT=${shaderSpv} -DOUTPUT=${shaderHeader} -DSYMBOL=${shaderSymbol} -P "${agcKytyRoot}/embed_spirv.cmake"
            DEPENDS "${shaderSource}" ${agcKytyTilerIncludes} glslang-standalone
            VERBATIM
    )
    list(APPEND agcKytyShaderHeaders "${shaderHeader}")
endforeach()

set(faultShaderSpv "${agcKytyGenerated}/gpu_tiler_shaders/fault_buffer_process.spv")
set(faultShaderHeader "${agcKytyGenerated}/gpu_tiler_shaders/fault_buffer_process_spv.h")
add_custom_command(
        OUTPUT "${faultShaderHeader}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${agcKytyGenerated}/gpu_tiler_shaders"
        COMMAND "$<TARGET_FILE:glslang-standalone>" -V --target-env vulkan1.3 -DCACHING_PAGEBITS=14 -DMAX_PAGE_FAULTS=1024 -o "${faultShaderSpv}" "${agcKytyShaders}/fault_buffer_process.comp"
        COMMAND ${CMAKE_COMMAND} -DINPUT=${faultShaderSpv} -DOUTPUT=${faultShaderHeader} -DSYMBOL=FAULT_BUFFER_PROCESS_SPV -P "${agcKytyRoot}/embed_spirv.cmake"
        DEPENDS "${agcKytyShaders}/fault_buffer_process.comp" glslang-standalone
        VERBATIM
)
list(APPEND agcKytyShaderHeaders "${faultShaderHeader}")

function(agc_kyty_blit_shader shaderName shaderFile)
    set(shaderSpv "${agcKytyGenerated}/gpu_blit_shaders/${shaderName}.spv")
    set(shaderHeader "${agcKytyGenerated}/gpu_blit_shaders/${shaderName}_spv.h")
    string(TOUPPER "${shaderName}_SPV" shaderSymbol)
    add_custom_command(
            OUTPUT "${shaderHeader}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${agcKytyGenerated}/gpu_blit_shaders"
            COMMAND "$<TARGET_FILE:glslang-standalone>" -V --target-env vulkan1.3 ${ARGN} "-I${agcKytyShaders}" -o "${shaderSpv}" "${agcKytyShaders}/${shaderFile}"
            COMMAND ${CMAKE_COMMAND} -DINPUT=${shaderSpv} -DOUTPUT=${shaderHeader} -DSYMBOL=${shaderSymbol} -P "${agcKytyRoot}/embed_spirv.cmake"
            DEPENDS "${agcKytyShaders}/${shaderFile}" "${agcKytyShaders}/gpu_video_out_hdr10_to_sdr.inc" glslang-standalone
            VERBATIM
    )
    set(agcKytyShaderHeaders ${agcKytyShaderHeaders} "${shaderHeader}" PARENT_SCOPE)
endfunction()
agc_kyty_blit_shader(gpu_blit_fs_triangle gpu_blit_fs_triangle.vert)
agc_kyty_blit_shader(gpu_blit_color_to_ms_depth gpu_blit_color_to_ms_depth.frag)
agc_kyty_blit_shader(gpu_pssr_linear gpu_pssr_linear.comp)
agc_kyty_blit_shader(gpu_bvh_mirror gpu_bvh_mirror.comp)
agc_kyty_blit_shader(gpu_dispatch_clamp gpu_dispatch_clamp.comp)
agc_kyty_blit_shader(gpu_bvh_walk gpu_bvh_mirror.comp -DWALK=1)
agc_kyty_blit_shader(gpu_mesh_indirect gpu_mesh_indirect.task --target-env spirv1.5)

add_library(agc_kyty_gpu STATIC EXCLUDE_FROM_ALL ${agcKytySources} ${agcKytyShaderHeaders})
target_include_directories(agc_kyty_gpu PRIVATE
        ${agcKytyRoot}/shim
        ${agcKytySource}
        ${agcKytyRoot}
        ${agcKytyGenerated}
        ${LIBS_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/core/libs
        ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR}
        ${ANYPS5_SPIRV_HEADERS_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/3rdparty/VulkanMemoryAllocator/include
        ${CMAKE_SOURCE_DIR}/3rdparty/fmt/include
        ${CMAKE_SOURCE_DIR}/3rdparty/magic_enum/include/magic_enum
        ${CMAKE_SOURCE_DIR}/3rdparty/xxHash
        ${CMAKE_SOURCE_DIR}/3rdparty/xbyak
        ${CMAKE_SOURCE_DIR}/3rdparty/tracy/public
)
target_include_directories(agc_kyty_gpu SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/zstd/lib)
target_compile_definitions(agc_kyty_gpu PRIVATE
        NOMINMAX
        FMT_HEADER_ONLY
        XXH_INLINE_ALL
        XBYAK_NO_EXCEPTION
        KYTY_HAS_SPIRV_TOOLS=0
)
if(WIN32)
    target_compile_options(agc_kyty_gpu PRIVATE "SHELL:-include \"${CMAKE_SOURCE_DIR}/core/libs/prx/libc/include/HostStandardStreams.hpp\"")
endif()
target_link_libraries(agc_kyty_gpu PRIVATE libc libzstd_static)
set_target_properties(agc_kyty_gpu PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
add_dependencies(agc_kyty_gpu glslang-standalone)

target_link_libraries(libSceAgcDriver PRIVATE agc_kyty_gpu)
