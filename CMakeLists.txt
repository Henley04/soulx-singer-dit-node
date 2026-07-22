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
# Exactly one backend is typically enabled per build; CI produces one .node
# per backend and the JS layer loads the right one.
# ---------------------------------------------------------------------------
option(GGML_CPU        "Enable ggml CPU backend"            ON)
option(GGML_VULKAN     "Enable ggml Vulkan backend"         OFF)
option(GGML_CUDA       "Enable ggml CUDA backend"           OFF)
option(GGML_METAL      "Enable ggml Metal backend (macOS)"  OFF)
option(GGML_ACCELERATE "Enable Accelerate framework (macOS)" OFF)
option(GGML_BLAS       "Use BLAS"                           OFF)
option(GGML_OPENMP     "Use OpenMP"                         ON)

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
        execute_process(
            COMMAND ${NODE_EXECUTABLE} -p
                "require('path').dirname(process.execPath) + '/include/node'"
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
message(STATUS "  GGML_CPU       = ${GGML_CPU}")
message(STATUS "  GGML_VULKAN    = ${GGML_VULKAN}")
message(STATUS "  GGML_CUDA      = ${GGML_CUDA}")
message(STATUS "  GGML_METAL     = ${GGML_METAL}")
message(STATUS "  GGML_ACCELERATE= ${GGML_ACCELERATE}")
message(STATUS "  GGML_BLAS      = ${GGML_BLAS}")
message(STATUS "  GGML_OPENMP    = ${GGML_OPENMP}")
message(STATUS "  LLAMA_CPP_DIR  = ${LLAMA_CPP_DIR}")
message(STATUS "  NODE_ADDON_API = ${NODE_ADDON_API_DIR}")
message(STATUS "  NODE_INCLUDE   = ${NODE_INCLUDE_DIRS}")

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
# Our native addon — a single .node shared library.
# ---------------------------------------------------------------------------
add_library(soulx_singer_dit SHARED
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

# Link ggml (pulls in ggml-base + whichever backends are enabled). No extra
# link line is needed for Node.js itself — N-API symbols resolve from the
# host node process at load time on every platform.
target_link_libraries(soulx_singer_dit PRIVATE ggml)

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
