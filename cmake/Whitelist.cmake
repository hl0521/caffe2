#
message(STATUS "  __caffe2_whitelist_included:  ${__caffe2_whitelist_included}")
if (__caffe2_whitelist_included)
  return()
endif()

set (__caffe2_whitelist_included TRUE)

set(CAFFE2_WHITELISTED_FILES)
message(STATUS "  CAFFE2_WHITELIST:  ${CAFFE2_WHITELIST}")
if (NOT CAFFE2_WHITELIST)
  return()
endif()

# First read the whitelist file and break it by line.
message(STATUS "  CAFFE2_WHITELIST:  ${CAFFE2_WHITELIST}")
file(READ "${CAFFE2_WHITELIST}" whitelist_content)
# Convert file contents into a CMake list
string(REGEX REPLACE "\n" ";" whitelist_content ${whitelist_content})
message(STATUS "  whitelist_content:  ${whitelist_content}")

foreach(item ${whitelist_content})
  file(GLOB_RECURSE tmp ${item})
  set(CAFFE2_WHITELISTED_FILES ${CAFFE2_WHITELISTED_FILES} ${tmp})
  message(STATUS "  CAFFE2_WHITELISTED_FILES:  ${CAFFE2_WHITELISTED_FILES}")
endforeach()

macro(caffe2_do_whitelist output whitelist)
  set(_tmp)
  foreach(item ${${output}})
    # 从 whitelist 列表中查找 item，如果找到，返回它的 _index；如果没找到，返回 -1
    list(FIND ${whitelist} ${item} _index)
    if (${_index} GREATER -1)
      set(_tmp ${_tmp} ${item})
    endif()
  endforeach()
  set(${output} ${_tmp})
endmacro()
