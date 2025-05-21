cmake_minimum_required(VERSION 3.1...3.29)

project(
	ExampleDecoder
	VERSION 1.0
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix fuse.h 32/64bit offset issues
add_compile_definitions(_FILE_OFFSET_BITS=64)

# ----------------------------------------------------------------
#  Platform‐specific: Linux/Mac us libfuse2 via pkg-config
# ----------------------------------------------------------------
if(APPLE)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -march=armv8-a -mcpu=apple-m1 -funroll-loops -flto")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a -mcpu=apple-m1 -funroll-loops -flto")

  # 1) Tell pkg-config where to look for .pc files
  set(ENV{PKG_CONFIG_PATH}
      "/usr/local/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")

  # 2) Find libfuse2
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBFUSE2 REQUIRED fuse)

  include_directories(
    ${LIBFUSE2_INCLUDE_DIRS}
    motioncam_decoder
    lib/include
    thirdparty
  )
  link_directories(${LIBFUSE2_LIBRARY_DIRS})

  # Our library
  add_library(motioncam_decoder
    lib/Decoder.cpp
    lib/RawData.cpp
    lib/RawData_Legacy.cpp
  )
  set_property(TARGET motioncam_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)

  # Our executable
  add_executable(mcraw-mounter-fuse mcraw-mounter-fuse.cpp)
  target_link_libraries(mcraw-mounter-fuse
    PRIVATE motioncam_decoder ${LIBFUSE2_LIBRARIES}
  )
elseif(WIN32)

  # 1) Force the “Program Files (x86)” install-root
  set(WINFSP_ROOT "$ENV{ProgramFiles\(x86\)}/WinFsp")

  # 2) Detect target architecture so we pick the correct .lib/.dll names
  set(WINFSP_ARCH a64)

  # 3) Point at include, lib and bin sub-dirs under the hard-coded root
  set(WINFSP_INCLUDE_DIR "${WINFSP_ROOT}/inc/fuse")
  set(WINFSP_LIB_DIR     "${WINFSP_ROOT}/lib")
  set(WINFSP_BIN_DIR     "${WINFSP_ROOT}/bin")

  # 4) Basic sanity checks
  if(NOT EXISTS "${WINFSP_LIB_DIR}/winfsp-${WINFSP_ARCH}.lib")
    message(FATAL_ERROR "Cannot find winfsp-${WINFSP_ARCH}.lib in ${WINFSP_LIB_DIR}")
  endif()
  if(NOT EXISTS "${WINFSP_BIN_DIR}/winfsp-${WINFSP_ARCH}.dll")
    message(FATAL_ERROR "Cannot find winfsp-${WINFSP_ARCH}.dll in ${WINFSP_BIN_DIR}")
  endif()

  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   /O2 /Ob2 /Ot /Oi /Gy /GL /arch:AVX2 /fp:fast")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Ot /Oi /Gy /GL /arch:AVX2 /fp:fast")

  set(CMAKE_EXE_LINKER_FLAGS_RELEASE   "${CMAKE_EXE_LINKER_FLAGS_RELEASE}   /LTCG")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")

  # 5) Create our decoder library
  include_directories(
    ${WINFSP_INCLUDE_DIR}
    motioncam_decoder
    lib/include
    thirdparty
  )
  add_library(motioncam_decoder
    lib/Decoder.cpp
    lib/RawData.cpp
    lib/RawData_Legacy.cpp
  )
  set_property(TARGET motioncam_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)

  # 6) Build our executable and link against WinFsp
  add_executable(mcraw-mounter-fuse mcraw-mounter-fuse.cpp)
  target_link_directories(mcraw-mounter-fuse PRIVATE ${WINFSP_LIB_DIR})
  target_link_libraries(mcraw-mounter-fuse
    PRIVATE
      motioncam_decoder
      winfsp-${WINFSP_ARCH}  # import library
      Delayimp              # required for /DELAYLOAD
  )
  target_link_options(mcraw-mounter-fuse PRIVATE
    "/DELAYLOAD:winfsp-${WINFSP_ARCH}.dll"
  )

  # 7) Copy the DLL next to the .exe
  add_custom_command(TARGET mcraw-mounter-fuse POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${WINFSP_BIN_DIR}/winfsp-${WINFSP_ARCH}.dll"
      $<TARGET_FILE_DIR:mcraw-mounter-fuse>
    COMMENT "Copying WinFsp DLL to output folder"
  )

else()
  message(FATAL_ERROR "Unsupported platform")
endif()