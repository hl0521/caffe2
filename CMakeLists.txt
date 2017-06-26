cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(Caffe2 CXX C)

# TODO(bwasti): versioning

# Useful functions.
# 从 INPUT 中移除 ARGN，然后赋值给 OUTPUT
function (exclude OUTPUT INPUT)
# message(STATUS "  function exclude:")
# message(STATUS "  OUTPUT:  ${OUTPUT}")
# message(STATUS "  INPUT:   ${INPUT}")
# message(STATUS "  ARGN:    ${ARGN}")
set(EXCLUDES ${ARGN})
foreach(EXCLUDE ${EXCLUDES})
        list(REMOVE_ITEM INPUT "${EXCLUDE}")
endforeach()
# message(STATUS "  INPUT:   ${INPUT}")
set(${OUTPUT} ${INPUT} PARENT_SCOPE)
# message(STATUS "  OUTPUT:  ${OUTPUT}")
endfunction(exclude)

# 将 ARGN 中的变量分别添加到 PREPEND 后面，然后赋值给 OUTPUT
function (prepend OUTPUT PREPEND)
message(STATUS "  function prepend:")
message(STATUS "  OUTPUT:  ${OUTPUT}")
message(STATUS "  PREPEND: ${PREPEND}")
message(STATUS "  ARGN:    ${ARGN}")
set(OUT "")
foreach(ITEM ${ARGN})
        list(APPEND OUT "${PREPEND}${ITEM}")
endforeach()
message(STATUS "  OUT:     ${OUT}")
set(${OUTPUT} ${OUT} PARENT_SCOPE)
message(STATUS "  OUTPUT:  ${OUTPUT}")
endfunction(prepend)

# ---[ CMake scripts + modules
# 设置当前工程的 Modules 路径，在后面调用 find_package 时，会从这个变量指定的路径中
# 查找相应的 Find<package>.cmake 文件
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(CMAKE_MACOSX_RPATH 1)

# 为当前及其下级目录打开测试功能
enable_testing()

# options
option(USE_THREADS "Use Threads" ON)
option(USE_NERVANA_GPU "Use Nervana GPU backend" OFF)
option(USE_GLOG "Use GLOG" ON)
option(USE_GFLAGS "Use GFLAGS" ON)
option(USE_LMDB "Use LMDB" ON)
option(USE_LEVELDB "Use LMDB" ON)
option(USE_LITE_PROTO "Use lite protobuf instead of full." OFF)
option(USE_NCCL "Use NCCL" ON)
option(USE_NNPACK "Use NNPACK" ON)
option(USE_OPENCV "Use openCV" ON)
option(USE_CUDA "Use Cuda" ON)
option(USE_CNMEM "Use CNMEM" OFF)
option(USE_ZMQ "Use ZMQ" OFF)
option(USE_ROCKSDB "Use RocksDB" ON)
option(USE_REDIS "Use Redis" OFF)
option(USE_MPI "Use MPI" ON)
option(USE_GLOO "Use Gloo" ON)
option(BUILD_SHARED_LIBS "Build libcaffe2.so" ON)
option(USE_OPENMP "Use OpenMP for parallel code" ON)
option(BUILD_PYTHON "Build Python binaries" ON)
option(BUILD_BINARY "Build C++ binaries" ON)

# External projects（目前看来，这个变量为空，没用）
message(STATUS "  ExternalProject:  ${ExternalProject}")
include(ExternalProject)

include(cmake/Utils.cmake)
include(cmake/Summary.cmake)

# options that do not affect the main binaries, but affects testing binaries
option(BUILD_TEST "Build C++ test binaries (need gtest and gbenchmark)" ON)

set(CAFFE2_CPU_FLAGS "" CACHE STRING "Flags to specify CPU features.")
set(CAFFE2_WHITELIST "" CACHE STRING "A whitelist file of files that one should build.")
message(STATUS "  CAFFE2_CPU_FLAGS:  ${CAFFE2_CPU_FLAGS}")
message(STATUS "  CAFFE2_WHITELIST:  ${CAFFE2_WHITELIST}")

# Set default build type
# CMAKE_BUILD_TYPE 的值为 Release，暂不知是何处赋值（或系统默认）？
dmsg("  CMAKE_BUILD_TYPE:  ${CMAKE_BUILD_TYPE}")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif()

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Misc checks to cope with various compiler modes
include(cmake/MiscCheck.cmake)

# ---[ Whitelist file if whitelist is specified
include(cmake/Whitelist.cmake)

# ---[ Set link flag, handle additional deps for gcc 4.8 and above
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.0 AND NOT ANDROID)
  message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION}: Adding gcc and gcc_s libs to link line")
  list(APPEND Caffe2_DEPENDENCY_LIBS gcc_s gcc)
endif()

# ---[ Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "binaries")

# ---[ Build flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
message(STATUS "  CMAKE_C_FLAGS:  ${CMAKE_C_FLAGS}")
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
else()
  if (NOT ${BUILD_SHARED_LIBS})
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
endif()

# Caffe2_DEPENDENCY_LIBS Caffe2的依赖库，包含了Caffe2依赖的所有库文件，它主要是在
# cmake/Dependencies.cmake 文件中完善的，主要是 .so 文件和 .a 文件
if(NOT APPLE AND UNIX)
  list(APPEND Caffe2_DEPENDENCY_LIBS dl)
endif()
message(STATUS "  Caffe2_DEPENDENCY_LIBS:  ${Caffe2_DEPENDENCY_LIBS}")

if (CAFFE2_CPU_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CAFFE2_CPU_FLAGS}")
endif()

# ---[ Include path needed for proto
message(STATUS "  PROJECT_BINARY_DIR:  ${PROJECT_BINARY_DIR}")
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# ---[ Third party builds.
message(STATUS "  PROJECT_BINARY_DIR:  ${PROJECT_BINARY_DIR}")
include_directories(${PROJECT_BINARY_DIR})

# ---[ Old caffe protobuf.
add_subdirectory(caffe/proto)

# ---[ Main build
add_subdirectory(caffe2)

Caffe2_print_configuration_summary()
