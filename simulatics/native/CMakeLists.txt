cmake_minimum_required(VERSION 3.16)
project(simulatics_native CXX)

# Every src/<name>.cpp becomes its own shared library bin/<name>.dll (.so on Linux).
# Separate libraries mean rebuilding one kernel never has to overwrite a DLL that a
# running Python process (the GUI, a test) has loaded - Windows locks loaded DLLs.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach(src ${KERNEL_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  add_library(${name} SHARED ${src})
  set_target_properties(${name} PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(MSVC)
    target_compile_options(${name} PRIVATE /O2 /fp:fast /W3 /openmp:llvm)
  else()
    target_compile_options(${name} PRIVATE -O3 -ffast-math -Wall)
  endif()
  foreach(cfg "" _RELEASE _DEBUG _RELWITHDEBINFO _MINSIZEREL)
    set_target_properties(${name} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY${cfg} "${CMAKE_CURRENT_SOURCE_DIR}/bin"
      LIBRARY_OUTPUT_DIRECTORY${cfg} "${CMAKE_CURRENT_SOURCE_DIR}/bin")
  endforeach()
endforeach()
