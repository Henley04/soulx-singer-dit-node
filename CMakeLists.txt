# CMakeLists.txt — Build the soulx_singer_dit Node.js native addon.
#
# Produces a single soulx_singer_dit.node file that CI renames per backend
# (soulx_singer_dit.cpu.node / .vulkan.node / .cuda.node).
#
# Typical usage with cmake-js:
#   npx cmake-js compile -GGGML_VULKAN=ON
# or with plain CMake:
#   cmake -S . -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp
#   cmake --build build --config Release
#
# The addon links ggml statically (BUILD_SHARED_LIBS=OFF), so the resulting
# .node is self-contained except for system shared libs (libomp, libcuda, etc.).

cmake_minimum_required(VERSION 3.18)
project(soulx_singer_dit_node LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Backend selection options (passed through to llama.cpp's ggml CMake).
#
# Distribution model: a single .node contains the CPU backend (always) plus
# whichever GPU backends are enabled here. ggml's backends are plugins and
# happily coexist in one binary; the C++ runtime enumerates them with
# ggml_backend_dev_count() and picks the best one (see pick_backend() in
# infer.cpp). So the SAME source produces:
#   - a CPU-only .node          (GGML_CPU=ON only)
#   - a CPU+CUDA+Vulkan .node   (all three ON)  -> "single artifact" per OS
#   - a CPU+Metal .node         (macOS)
#
# CRITICAL: GGML_NATIVE is forced OFF below so the compiler does NOT emit
# -march=native. With -march=native the binary inherits the build machine's
# ISA and SIGILLs on older CPUs. Instead we compile a baseline ISA (SSE2/AVX
# on x86, NEON on ARM) plus the AMX/AVX512/AVX2 kernels as separate object
# files; ggml-cpu's mul_mat dispatch picks the right kernel at runtime via
# ggml_cpu_has_*(). AMX kernels are compiled in (GGML_AMX_*) but only
# executed when ggml_cpu_has_amx_int8() is true at runtime, so they never
# SIGILL on non-AMX hardware.
# ---------------------------------------------------------------------------
option(GGML_CPU        "Enable ggml CPU backend"            ON)
option(GGML_VULKAN     "Enable ggml Vulkan backend"         OFF)
option(GGML_CUDA       "Enable ggml CUDA backend"           OFF)
option(GGML_HIP        "Enable ggml HIP backend (AMD)"      OFF)
option(GGML_SYCL       "Enable ggml SYCL backend (Intel)"   OFF)
option(GGML_METAL      "Enable ggml Metal backend (macOS)"  OFF)
option(GGML_ACCELERATE "Enable Accelerate framework (macOS)" OFF)
option(GGML_BLAS       "Use BLAS"                           OFF)
option(GGML_OPENMP     "Use OpenMP"                         ON)

# Force the distribution-safe CPU configuration. These are ggml cache
# variables set BEFORE add_subdirectory(llama.cpp) so llama.cpp's option()
# calls honour them.
set(GGML_NATIVE OFF CACHE BOOL "Disable -march=native; rely on ggml runtime dispatch" FORCE)
set(GGML_LTO    OFF  CACHE BOOL "Disable LTO for reproducible dispatch" FORCE)

# AMX: compile the kernels in (x86 only) so they are available when the host
# CPU supports them. ggml-cpu guards execution with ggml_cpu_has_amx_int8().
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64|i386|i686)")
    set(GGML_AMX_TILE ON  CACHE BOOL "Compile AMX tile kernels (runtime-gated)" FORCE)
    set(GGML_AMX_INT8 ON  CACHE BOOL "Compile AMX int8 kernels (runtime-gated)" FORCE)
    set(GGML_AMX_BF16 ON  CACHE BOOL "Compile AMX bf16 kernels (runtime-gated)" FORCE)
else()
    set(GGML_AMX_TILE OFF CACHE BOOL "" FORCE)
    set(GGML_AMX_INT8 OFF CACHE BOOL "" FORCE)
    set(GGML_AMX_BF16 OFF CACHE BOOL "" FORCE)
endif()

# NUMA awareness on Linux: ggml-cpu reads the NUMA topology at backend init
# and pins worker threads per-socket. Avoids cross-socket memory traffic that
# halves AMX/AVX512 throughput on multi-socket boxes.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(GGML_NUMA ON CACHE BOOL "Enable NUMA awareness" FORCE)
endif()

# ARM: expose NEON/i8mm/SVE capable baseline for aarch64 (Graviton, Apple
# Silicon, Snapdragon). ggml-cpu runtime-dispatches among NEON/i8mm/SVE.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
    if(NOT DEFINED GGML_ARM_ARCH)
        set(GGML_ARM_ARCH "armv8.2-a+fp16+dotprod+i8mm" CACHE STRING "ARM baseline ISA" FORCE)
    endif()
endif()

# CUDA graphs: capture the per-step graph once and replay it, eliminating
# per-kernel launch overhead (22 layers x dozens of kernels per step).
#
# CRITICAL: restrict the target CUDA architectures. llama.cpp's default
# (GGML_CUDA_ALL_ARCHS=ON) compiles every ggml CUDA kernel (mmvq/mmq/conv/
# norm/...) for ~10 compute capabilities (6.0 through 9.0). On a 4-core CI
# runner nvcc takes 5+ hours and blows past the GitHub Actions 6h job limit.
# We target only modern consumer/datacenter GPUs so the build finishes in a
# few minutes; older GPUs (Pascal/Volta) fall back to the CPU backend at
# runtime. Override locally with -DCMAKE_CUDA_ARCHITECTURES=... if needed.
#   75 = Turing  (RTX 20xx, T4)
#   80 = Ampere  (A100)
#   86 = Ampere  (RTX 30xx, A10/A40)
#   89 = Ada     (RTX 40xx, L4)
if(GGML_CUDA)
    set(GGML_CUDA_USE_GRAPHS ON CACHE BOOL "Use CUDA graph capture" FORCE)
    set(GGML_CUDA_ALL_ARCHS OFF CACHE BOOL "Build only the arch subset below" FORCE)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "Target CUDA architectures" FORCE)
endif()

# ---------------------------------------------------------------------------
# Locate llama.cpp source (expected at vendor/llama.cpp).
# ---------------------------------------------------------------------------
set(LLAMA_CPP_DIR ${CMAKE_SOURCE_DIR}/vendor/llama.cpp CACHE PATH
    "Path to llama.cpp source tree (must contain CMakeLists.txt)")
if(NOT EXISTS ${LLAMA_CPP_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
        "llama.cpp source not found at '${LLAMA_CPP_DIR}'.\n"
        "Populate it first, e.g.:\n"
        "  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp\n"
        "or pass -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

# ---------------------------------------------------------------------------
# Locate node-addon-api (expected at node_modules/node-addon-api).
# ---------------------------------------------------------------------------
set(NODE_ADDON_API_DIR ${CMAKE_SOURCE_DIR}/node_modules/node-addon-api CACHE PATH
    "Path to node-addon-api (must contain napi.h)")
if(NOT EXISTS ${NODE_ADDON_API_DIR}/napi.h)
    message(FATAL_ERROR
        "node-addon-api not found at '${NODE_ADDON_API_DIR}'.\n"
        "Run `npm install` first, or pass -DNODE_ADDON_API_DIR=/path/to/node-addon-api")
endif()

# ---------------------------------------------------------------------------
# Locate Node.js headers.
#   cmake-js exposes them via CMAKE_JS_INC. Otherwise we try to derive them
#   from the `node` executable. The user can also pass -DNODE_INCLUDE_DIR=...
#   or set the NODE_INCLUDE_DIR environment variable.
# ---------------------------------------------------------------------------
if(DEFINED CMAKE_JS_INC)
    set(NODE_INCLUDE_DIRS ${CMAKE_JS_INC})
elseif(DEFINED ENV{NODE_INCLUDE_DIR})
    set(NODE_INCLUDE_DIRS $ENV{NODE_INCLUDE_DIR})
elseif(NOT DEFINED NODE_INCLUDE_DIRS)
    find_program(NODE_EXECUTABLE node)
    if(NODE_EXECUTABLE)
        # Ask node where its headers live for the current installation.
        # process.execPath is <prefix>/bin/node, so the headers ship at
        # <prefix>/include/node (NOT <prefix>/bin/include/node). Go up one
        # dir from bin/ to reach the prefix. This matters for nvm and for
        # actions/setup-node distributions on Linux/macOS runners.
        execute_process(
            COMMAND ${NODE_EXECUTABLE} -p
                "require('path').join(require('path').dirname(require('path').dirname(process.execPath)), 'include', 'node')"
            OUTPUT_VARIABLE _node_inc
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _node_inc_rc
        )
        if(_node_inc_rc EQUAL 0 AND EXISTS "${_node_inc}/node_api.h")
            set(NODE_INCLUDE_DIRS "${_node_inc}")
        endif()
    endif()
endif()

if(NOT NODE_INCLUDE_DIRS)
    message(WARNING
        "Node.js headers not found automatically.\n"
        "Build with cmake-js (recommended), or pass "
        "-DNODE_INCLUDE_DIRS=/path/to/node/include/node")
    set(NODE_INCLUDE_DIRS "")
endif()

message(STATUS "soulx_singer_dit configuration:")
message(STATUS "  GGML_CPU         = ${GGML_CPU}")
message(STATUS "  GGML_VULKAN      = ${GGML_VULKAN}")
message(STATUS "  GGML_CUDA        = ${GGML_CUDA}")
message(STATUS "  GGML_HIP         = ${GGML_HIP}")
message(STATUS "  GGML_SYCL        = ${GGML_SYCL}")
message(STATUS "  GGML_METAL       = ${GGML_METAL}")
message(STATUS "  GGML_ACCELERATE  = ${GGML_ACCELERATE}")
message(STATUS "  GGML_BLAS        = ${GGML_BLAS}")
message(STATUS "  GGML_OPENMP      = ${GGML_OPENMP}")
message(STATUS "  GGML_NATIVE      = ${GGML_NATIVE}   (OFF => no -march=native, runtime dispatch)")
message(STATUS "  GGML_AMX_TILE    = ${GGML_AMX_TILE}")
message(STATUS "  GGML_AMX_INT8    = ${GGML_AMX_INT8}")
message(STATUS "  GGML_AMX_BF16    = ${GGML_AMX_BF16}")
message(STATUS "  GGML_NUMA        = ${GGML_NUMA}")
message(STATUS "  GGML_ARM_ARCH    = ${GGML_ARM_ARCH}")
message(STATUS "  GGML_CUDA_GRAPHS = ${GGML_CUDA_USE_GRAPHS}")
message(STATUS "  LLAMA_CPP_DIR    = ${LLAMA_CPP_DIR}")
message(STATUS "  NODE_ADDON_API   = ${NODE_ADDON_API_DIR}")
message(STATUS "  NODE_INCLUDE     = ${NODE_INCLUDE_DIRS}")

# ---------------------------------------------------------------------------
# Configure llama.cpp before add_subdirectory.
#   - Build ggml (and only ggml) statically into the .node.
#   - Disable all of llama.cpp's own examples/tests/tools — we only need libggml.
#   - The GGML_* cache variables below are honoured by llama.cpp's option()
#     calls because we set them before add_subdirectory.
# ---------------------------------------------------------------------------
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libs" FORCE)

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama_cpp_build EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Our native addon — a single .node loadable module.
#
# MODULE (not SHARED) is the correct library type for a Node native addon:
#   - macOS: produces a -bundle (MACH-O MH_BUNDLE) which the Node process
#     dlopen()s. SHARED would produce a -dynamiclib that the macOS linker
#     refuses to leave with unresolved _napi_* symbols at link time. MODULE
#     + the -undefined dynamic_lookup flag below lets those symbols resolve
#     from the host Node process at runtime, matching Linux/Windows behaviour.
#   - Linux: MODULE is equivalent to SHARED for our purposes (the .node is
#     dlopen'd by Node), and CMake leaves undefined symbols by default.
#   - Windows: MODULE still produces a .dll; the explicit node.lib link
#     below resolves the N-API symbols at link time as before.
# ---------------------------------------------------------------------------
add_library(soulx_singer_dit MODULE
    src/binding.cc
    src/infer.cpp
)

set_target_properties(soulx_singer_dit PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    OUTPUT_NAME "soulx_singer_dit"   # CI renames to cpu/vulkan/cuda
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(soulx_singer_dit PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${NODE_ADDON_API_DIR}
    ${NODE_INCLUDE_DIRS}
)

# node-addon-api expects C++ exceptions by default (NAPI_CPP_EXCEPTIONS).
target_compile_definitions(soulx_singer_dit PRIVATE NAPI_CPP_EXCEPTIONS)

if(MSVC)
    target_compile_options(soulx_singer_dit PRIVATE /EHsc /utf-8 /bigobj)
else()
    target_compile_options(soulx_singer_dit PRIVATE -fexceptions)
endif()

# Link ggml (pulls in ggml-base + whichever backends are enabled).
target_link_libraries(soulx_singer_dit PRIVATE ggml)

# macOS: N-API symbols (_napi_*) are provided by the host Node process at
# runtime, but the macOS linker rejects unresolved symbols by default.
# -undefined dynamic_lookup defers their resolution to dlopen() time, so the
# .node bundle loads cleanly when Node requires() it. Linux leaves undefined
# symbols alone already; Windows resolves them via node.lib below.
if(APPLE)
    target_link_options(soulx_singer_dit PRIVATE -undefined dynamic_lookup)
endif()

# On Windows, N-API symbols do NOT resolve from the host node process at
# link time — unlike Linux/macOS, the Windows linker requires an explicit
# import library (node.lib).  We look for it via NODE_LIB_DIR (set by CI)
# or relative to the Node headers tree.
if(WIN32)
    # Try NODE_LIB_DIR environment variable first (most reliable, set by CI).
    if(DEFINED ENV{NODE_LIB_DIR})
        find_library(NODE_LIB node PATHS "$ENV{NODE_LIB_DIR}" NO_DEFAULT_PATH)
    endif()
    # Fall back: search relative to the include directory.
    if(NOT NODE_LIB AND NODE_INCLUDE_DIRS)
        get_filename_component(_inc_dir "${NODE_INCLUDE_DIRS}" PATH)        # .../include
        get_filename_component(_inc_root "${_inc_dir}" PATH)               # .../node-headers
        find_library(NODE_LIB node PATHS "${_inc_root}/lib" "${_inc_dir}/lib" NO_DEFAULT_PATH)
    endif()
    if(NODE_LIB)
        target_link_libraries(soulx_singer_dit PRIVATE ${NODE_LIB})
        message(STATUS "  NODE_LIB       = ${NODE_LIB}")
    else()
        message(FATAL_ERROR
            "node.lib not found on Windows.\n"
            "Set NODE_LIB_DIR env var to the directory containing node.lib, or\n"
            "download it from https://nodejs.org/dist/<version>/win-x64/node.lib")
    endif()
endif()

# ---------------------------------------------------------------------------
# Installation: drop the .node file somewhere predictable. cmake-js usually
# handles placement itself, but this makes a plain `cmake --install` usable.
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS soulx_singer_dit
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
