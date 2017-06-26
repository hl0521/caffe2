# ---[ Declare source file lists

# Caffe2_{CPU,GPU}_SRCS is the list that will have all the related source
# files for CPU and GPU respectively. They will be filled with the
# CMakeLists.txt files under each folder respectively.
set(Caffe2_CPU_SRCS)
set(Caffe2_GPU_SRCS)

# Caffe2_{CPU,GPU}_TEST_SRCS is the list that will have all the related source
# files for CPU and GPU tests respectively.
set(Caffe2_CPU_TEST_SRCS)
set(Caffe2_GPU_TEST_SRCS)

# ---[ Add respective subdirectories
# Note: the folders that are being commented out have not been properly
# addressed yet.
# 包含根目录，用于查找头文件，否则在 Android 下会发生找不到头文件的错误
# 在 PC 环境下，不会发生，不需要下面这句话，原因不明？？？
include_directories(../)

# 下面这1-15，将 caffe2 目录下的文件进行分类：分别 CPU 源文件；GPU源文件
# CPU 测试文件和 GPU 测试文件
add_subdirectory(proto)
message(STATUS "  1-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")

add_subdirectory(contrib)
message(STATUS "  2-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(core)
message(STATUS "  3-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(cuda_rtc)
message(STATUS "  4-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(db)
message(STATUS "  5-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(distributed)
# add_subdirectory(experiments) # note, we may remove this folder at some point
message(STATUS "  6-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(image)
message(STATUS "  7-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(video)
message(STATUS "  8-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(mkl)
message(STATUS "  9-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(mpi)
message(STATUS "  10-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(operators)
message(STATUS "  11-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(python)
message(STATUS "  12-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(queue)
message(STATUS "  13-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(sgd)
# add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
message(STATUS "  14-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")
add_subdirectory(utils)
message(STATUS "  15-Caffe2_CPU_SRCS:  ${Caffe2_CPU_SRCS}")

# Advanced: if we have white list specified, we will do intersections for all
# main lib srcs.
# 这个地方做啥，暂时没有看懂？？？
message(STATUS "  CAFFE2_WHITELISTED_FILES:  ${CAFFE2_WHITELISTED_FILES}")
if (CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_CPU_SRCS CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_GPU_SRCS CAFFE2_WHITELISTED_FILES)
endif()

# Debug messages - if you want to get a list of source files, enable the
# following.
# true or false
if (true)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: ")
  foreach(tmp ${Caffe2_GPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU test sources: ")
  foreach(tmp ${Caffe2_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU test sources: ")
  foreach(tmp ${Caffe2_GPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

# ---[ List of libraries to link with

# In the static linking + clang mode, cmake will fail to identify the build
# order because the lib becomes one single string -Wl,-force-load,libCaffe2_CPU.so
# As a result, we will create a Caffe2_MAIN_LIBS_ORDER variable simply to
# enforce the dependency.
set(Caffe2_MAIN_LIBS_ORDER ${Caffe2_MAIN_LIBS})
message(STATUS "  Caffe2_MAIN_LIBS_ORDER:  ${Caffe2_MAIN_LIBS_ORDER}")

# Compile exposed libraries.
message(STATUS "  TARGET_OBJECTS:Caffe_PROTO: $<TARGET_OBJECTS:Caffe_PROTO>")
message(STATUS "  TARGET_OBJECTS:Caffe2_PROTO: $<TARGET_OBJECTS:Caffe2_PROTO>")
# 添加一个名为 Caffe2_CPU 的库文件，它根据 Caffe2_CPU 后列出的源文件来创建
# 后面 尖括号 中是几个意思？？？没太看懂，上面的调试输出语句也是原样输出
add_library(Caffe2_CPU ${Caffe2_CPU_SRCS} $<TARGET_OBJECTS:Caffe_PROTO> $<TARGET_OBJECTS:Caffe2_PROTO>)
message(STATUS "  Caffe2_DEPENDENCY_LIBS:  ${Caffe2_DEPENDENCY_LIBS}")
target_link_libraries(Caffe2_CPU ${Caffe2_DEPENDENCY_LIBS})
message(STATUS "  CMAKE_INSTALL_PREFIX:  ${CMAKE_INSTALL_PREFIX}")
install(TARGETS Caffe2_CPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
message(STATUS "  Caffe2_MAIN_LIBS_ORDER:  ${Caffe2_MAIN_LIBS_ORDER}")
list(APPEND Caffe2_MAIN_LIBS_ORDER Caffe2_CPU)
message(STATUS "  Caffe2_MAIN_LIBS_ORDER:  ${Caffe2_MAIN_LIBS_ORDER}")
if (BUILD_SHARED_LIBS)
  list(APPEND Caffe2_MAIN_LIBS Caffe2_CPU)
else()
  caffe_add_whole_archive_flag(Caffe2_CPU tmp)
  list(APPEND Caffe2_MAIN_LIBS ${tmp})
endif()
message(STATUS "  Caffe2_EXTERNAL_DEPENDENCIES:  ${Caffe2_EXTERNAL_DEPENDENCIES}")
if (Caffe2_EXTERNAL_DEPENDENCIES)
  add_dependencies(Caffe2_CPU ${Caffe2_EXTERNAL_DEPENDENCIES})
endif()

# Write the macros file.
message(STATUS "  PROJECT_BINARY_DIR:  ${PROJECT_BINARY_DIR}")
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/caffe2/core)
file(WRITE ${PROJECT_BINARY_DIR}/caffe2/core/macros.h
     "// Automatically generated header file for caffe2 macros. These \n"
     "// macros are used to build the Caffe2 binary, and if you are \n"
     "// building a dependent library, they will need to be set as well \n"
     "// for your program to link correctly.\n\n"
     "#pragma once\n\n")
get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
message(STATUS "  tmp:  ${tmp}")
foreach(item ${tmp})
  if (${item} MATCHES "CAFFE2.*")
    file(APPEND ${PROJECT_BINARY_DIR}/caffe2/core/macros.h
         "\n#ifndef ${item}\n#define ${item}\n#endif // ${item}\n")
  endif()
endforeach()

# Installing the header files
message(STATUS "  CMAKE_CURRENT_LIST_DIR:  ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "  CMAKE_INSTALL_PREFIX:  ${CMAKE_INSTALL_PREFIX}")
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
        FILES_MATCHING PATTERN "*.h")
install(FILES ${PROJECT_BINARY_DIR}/caffe2/core/macros.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/caffe2/core/)

# ---[ CUDA library.
if(USE_CUDA)
  CUDA_ADD_LIBRARY(Caffe2_GPU ${Caffe2_GPU_SRCS})
  list(APPEND Caffe2_MAIN_LIBS_ORDER Caffe2_GPU)
  if (BUILD_SHARED_LIBS)
    target_link_libraries(Caffe2_GPU Caffe2_CPU ${Caffe2_DEPENDENCY_LIBS})
    list(APPEND Caffe2_MAIN_LIBS Caffe2_GPU)
  else()
    target_link_libraries(Caffe2_GPU ${Caffe2_DEPENDENCY_LIBS})
    caffe_add_whole_archive_flag(Caffe2_GPU tmp)
    list(APPEND Caffe2_MAIN_LIBS ${tmp})
  endif()
  if (Caffe2_EXTERNAL_DEPENDENCIES)
    add_dependencies(Caffe2_GPU ${Caffe2_EXTERNAL_DEPENDENCIES})
  endif()
  install(TARGETS Caffe2_GPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  link_directories(${CMAKE_INSTALL_PREFIX}/lib)
endif()

# ---[ Test binaries.
message(STATUS "  BUILD_TEST:  ${BUILD_TEST}")
if (BUILD_TEST)
  set(Caffe2_ALL_TEST_SRCS ${Caffe2_CPU_TEST_SRCS})
  if (USE_CUDA)
    list(APPEND Caffe2_ALL_TEST_SRCS ${Caffe2_GPU_TEST_SRCS})
  endif()

  foreach(test_src ${Caffe2_ALL_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    add_dependencies(${test_name} ${Caffe2_MAIN_LIBS_ORDER})
    target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS} gtest_main)
    if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
      target_compile_features(${test_name} PRIVATE cxx_range_for)
    endif()
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    install(TARGETS ${test_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
  endforeach()
endif()


if (BUILD_PYTHON)
  # ---[ Python.
  add_library(caffe2_pybind11_state MODULE ${Caffe2_CPU_PYTHON_SRCS})
  add_dependencies(caffe2_pybind11_state ${Caffe2_MAIN_LIBS_ORDER})
  set_target_properties(caffe2_pybind11_state PROPERTIES PREFIX "")
  if (APPLE)
    set_target_properties(caffe2_pybind11_state PROPERTIES SUFFIX ".so")
  elseif (MSVC)
    set_target_properties(caffe2_pybind11_state PROPERTIES SUFFIX ".pyd")
  endif()
  set_target_properties(
      caffe2_pybind11_state PROPERTIES LIBRARY_OUTPUT_DIRECTORY
      ${CMAKE_BINARY_DIR}/caffe2/python)
  target_link_libraries(
      caffe2_pybind11_state ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS}
      ${Caffe2_PYTHON_DEPENDENCY_LIBS})
  # Add rpath to the python target, so that we will be able to load it. This
  # is in order to ease the problem of anaconda's ipython DYLD bug, detailed
  # here:
  #     https://github.com/BVLC/caffe/issues/2720
  #     https://github.com/ipython/ipython/issues/8878
  if (APPLE)
    add_custom_command(
        TARGET caffe2_pybind11_state POST_BUILD
        COMMAND install_name_tool -add_rpath ${CMAKE_INSTALL_PREFIX}/lib
                $<TARGET_FILE:caffe2_pybind11_state>)
  endif()
  install(
      TARGETS caffe2_pybind11_state DESTINATION
      ${CMAKE_INSTALL_PREFIX}/caffe2/python)

  if(USE_CUDA)
    add_library(caffe2_pybind11_state_gpu MODULE ${Caffe2_GPU_PYTHON_SRCS})
    add_dependencies(caffe2_pybind11_state_gpu ${Caffe2_MAIN_LIBS_ORDER})
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES PREFIX "")
    if (APPLE)
      set_target_properties(caffe2_pybind11_state_gpu PROPERTIES SUFFIX ".so")
    elseif (MSVC)
      set_target_properties(caffe2_pybind11_state_gpu PROPERTIES SUFFIX ".pyd")
    endif()
    set_target_properties(
        caffe2_pybind11_state_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/caffe2/python)
    target_link_libraries(
        caffe2_pybind11_state_gpu ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS}
        ${Caffe2_PYTHON_DEPENDENCY_LIBS})
    if (APPLE)
      add_custom_command(
          TARGET caffe2_pybind11_state_gpu POST_BUILD
          COMMAND install_name_tool -add_rpath ${CMAKE_INSTALL_PREFIX}/lib
                  $<TARGET_FILE:caffe2_pybind11_state_gpu>)
    endif()
    install(
        TARGETS caffe2_pybind11_state_gpu DESTINATION
        ${CMAKE_INSTALL_PREFIX}/caffe2/python)
  endif()

  if (MSVC)
    # If we are building under windows, we will copy the file from
    # build/caffe2/python/{Debug,Release}/caffe2_pybind11_state.pyd
    # to its parent folder so that we can do in-build execution.
    add_custom_target(windows_python_copy_lib ALL)
    add_dependencies(windows_python_copy_lib caffe2_pybind11_state)
    add_custom_command(
        TARGET windows_python_copy_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/caffe2/python/${CMAKE_BUILD_TYPE}/caffe2_pybind11_state.pyd
        ${CMAKE_BINARY_DIR}/caffe2/python)
    if (USE_CUDA)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_gpu)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_BINARY_DIR}/caffe2/python/${CMAKE_BUILD_TYPE}/caffe2_pybind11_state_gpu.pyd
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
  endif()

  # Finally, Copy all python files to build directory
  # Generate and create all needed __init__.py files, if they aren't already
  # present in the current source tree.
  message(STATUS "Automatically generating missing __init__.py files.")
  caffe_autogen_init_py_files()

  # Create a custom target that copies all python files.
  file(GLOB_RECURSE PYTHON_SRCS RELATIVE ${PROJECT_SOURCE_DIR}
       "${PROJECT_SOURCE_DIR}/caffe2/*.py")
  add_custom_target(python_copy_files ALL)
  foreach(python_src ${PYTHON_SRCS})
    get_filename_component(dir ${python_src} DIRECTORY)
    add_custom_command(
        TARGET python_copy_files PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${PROJECT_SOURCE_DIR}/${python_src} ${CMAKE_BINARY_DIR}/${dir})
  endforeach()

  # Install commands
  # Pick up static python files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${CMAKE_INSTALL_PREFIX}
          FILES_MATCHING PATTERN "*.py")
  # Caffe proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe DESTINATION ${CMAKE_INSTALL_PREFIX}
          FILES_MATCHING PATTERN "*.py")
  # Caffe2 proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${CMAKE_INSTALL_PREFIX}
          FILES_MATCHING PATTERN "*.py")
endif()


# ---[ Any builds that should happen after the main targets should be added here.
# Binaries
if (BUILD_BINARY)
  add_subdirectory(binaries)
endif()
