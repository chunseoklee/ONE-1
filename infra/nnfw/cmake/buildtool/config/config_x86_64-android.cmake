#
# x86_64 linux compile options
#
message(STATUS "Building for x86-64 Linux")

# include linux common
include("cmake/buildtool/config/config_linux.cmake")

# On Android, pthread is contained in bionic(libc)
set(LIB_PTHREAD "")

# SIMD for x86
set(FLAGS_COMMON ${FLAGS_COMMON}
    "-msse4"
    )
