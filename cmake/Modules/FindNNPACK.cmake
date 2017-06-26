# - Try to find NNPACK
#
# The following variables are optionally searched for defaults
#  NNPACK_ROOT_DIR:            Base directory where all NNPACK components are found
#
# The following are set after configuration is done:
#  NNPACK_FOUND
#  NNPACK_INCLUDE_DIRS
#  NNPACK_LIBRARIES
#  NNPACK_LIBRARYRARY_DIRS

# FindPackageHandleStandardArgs.cmake 文件在 /usr/share/cmake-2.8/Modules/ 文件夹下
# 里面定义了许多函数，具体参数该文件
include(FindPackageHandleStandardArgs)

set(NNPACK_ROOT_DIR "" CACHE PATH "Folder contains NNPACK")

message(STATUS "  NNPACK_ROOT_DIR:     ${NNPACK_ROOT_DIR}")
find_path(NNPACK_INCLUDE_DIR nnpack.h
    PATHS ${NNPACK_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(NNPACK_LIBRARY nnpack
    PATHS ${NNPACK_ROOT_DIR}
    PATH_SUFFIXES lib lib64)

message(STATUS "  NNPACK_LIBRARY:      ${NNPACK_LIBRARY}")
message(STATUS "  NNPACK_INCLUDE_DIR:  ${NNPACK_INCLUDE_DIR}")

find_package_handle_standard_args(NNPACK DEFAULT_MSG NNPACK_INCLUDE_DIR NNPACK_LIBRARY)

if(NNPACK_FOUND)
  set(NNPACK_INCLUDE_DIRS ${NNPACK_INCLUDE_DIR})
  set(NNPACK_LIBRARIES ${NNPACK_LIBRARY})
  message(STATUS "Found NNPACK    (include: ${NNPACK_INCLUDE_DIR}, library: ${NNPACK_LIBRARY})")
  mark_as_advanced(NNPACK_ROOT_DIR NNPACK_LIBRARY_RELEASE NNPACK_LIBRARY_DEBUG
                                 NNPACK_LIBRARY NNPACK_INCLUDE_DIR)
endif()
