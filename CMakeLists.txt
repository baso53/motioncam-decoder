cmake_minimum_required(VERSION 3.1...3.29)

project(ExampleDecoder
  VERSION 1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD      20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 64-bit file offsets everywhere
add_compile_definitions(_FILE_OFFSET_BITS=64)

# our shared decoder sources & common include dirs
set(DECODER_SRCS
  lib/Decoder.cpp
  lib/RawData.cpp
  lib/RawData_Legacy.cpp
)
set(COMMON_INCLUDES
  motioncam_decoder
  lib/include
  thirdparty
)

if(APPLE)
  # maximum optimization + LTO on macOS
  add_compile_options(-O3 -march=armv8-a -mcpu=apple-m1 -funroll-loops -flto)

  # find libfuse2 via pkg-config
  set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FUSE REQUIRED fuse)

  set(FUSE_INCLUDE_DIRS ${FUSE_INCLUDE_DIRS})
  set(FUSE_LIBS         ${FUSE_LIBRARIES})
  link_directories(${FUSE_LIBRARY_DIRS})

elseif(WIN32)
  # hard-coded WinFsp layout
  set(WINFSP_ROOT "C:/Program Files (x86)/WinFsp")
  set(WINFSP_ARCH a64)
  set(WINFSP_INC  "${WINFSP_ROOT}/inc/fuse")
  set(WINFSP_LIB  "${WINFSP_ROOT}/lib")
  set(WINFSP_BIN  "${WINFSP_ROOT}/bin")

  # sanity checks
  if(NOT EXISTS "${WINFSP_LIB}/winfsp-${WINFSP_ARCH}.lib")
    message(FATAL_ERROR "Cannot find winfsp-${WINFSP_ARCH}.lib in ${WINFSP_LIB}")
  endif()
  if(NOT EXISTS "${WINFSP_BIN}/winfsp-${WINFSP_ARCH}.dll")
    message(FATAL_ERROR "Cannot find winfsp-${WINFSP_ARCH}.dll in ${WINFSP_BIN}")
  endif()

  link_directories(${WINFSP_LIB})
  set(FUSE_INCLUDE_DIRS ${WINFSP_INC})
  set(FUSE_LIBS
    winfsp-${WINFSP_ARCH}  # import-lib
    Delayimp               # for /DELAYLOAD
  )

else()
  message(FATAL_ERROR "Unsupported platform")
endif()

# common includes
include_directories(${COMMON_INCLUDES} ${FUSE_INCLUDE_DIRS})

# build our decoder library
add_library(motioncam_decoder ${DECODER_SRCS})
set_property(TARGET motioncam_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)

# build & link the FUSE mount executable
add_executable(mcraw-mounter-fuse mcraw-mounter-fuse.cpp)
target_link_libraries(mcraw-mounter-fuse
  PRIVATE
    motioncam_decoder
    ${FUSE_LIBS}
)

if(WIN32)
    # 7) Copy the DLL next to the .exe
  add_custom_command(TARGET mcraw-mounter-fuse POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${WINFSP_BIN}/winfsp-${WINFSP_ARCH}.dll"
      $<TARGET_FILE_DIR:mcraw-mounter-fuse>
    COMMENT "Copying WinFsp DLL to output folder"
  )
endif()