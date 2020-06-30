# Reference: https://github.com/tensorflow/tensorflow/blob/v2.3.0-rc0/tensorflow/lite/tools/make/Makefile
#
# Tensorflow Lite library 2.3.0-rc0
#
set(TENSORFLOW_LITE_BASE ${TFLite230TensorFlowSource_DIR}/tensorflow/lite)

file(GLOB TFLITE_CORE_SRCS "${TENSORFLOW_LITE_BASE}/*.c"
     "${TENSORFLOW_LITE_BASE}/*.cc"
     "${TENSORFLOW_LITE_BASE}/core/*.cc")

file(GLOB_RECURSE TFLITE_KERNEL_SRCS "${TENSORFLOW_LITE_BASE}/kernels/*.cc")

file(GLOB TFLITE_LIB_SRCS "${TENSORFLOW_LITE_BASE}/c/*.c" "${TENSORFLOW_LITE_BASE}/c/*.cc")

file(GLOB TFLITE_API_SRCS "${TENSORFLOW_LITE_BASE}/core/api/*.c"
     "${TENSORFLOW_LITE_BASE}/core/api/*.cc")

list(APPEND TFLITE_PROFILING_SRCS "${TENSORFLOW_LITE_BASE}/profiling/memory_info.cc")
list(APPEND TFLITE_PROFILING_SRCS "${TENSORFLOW_LITE_BASE}/profiling/time.cc")

#file(GLOB TFLITE_EXPERIMENTAL_SRCS "${TENSORFLOW_LITE_BASE}/experimental/resource/*.cc"
# "${TENSORFLOW_LITE_BASE}/experimental/ruy/*.cc")

file(GLOB TFLITE_SPARSITY_SRCS "${TENSORFLOW_LITE_BASE}/tools/optimize/sparsity/*.cc")

list(APPEND TFLITE_SRCS ${TFLITE_CORE_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_KERNEL_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_LIB_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_API_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_PROFILING_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_EXPERIMENTAL_SRCS})
list(APPEND TFLITE_SRCS ${TFLITE_SPARSITY_SRCS})

# externals
list(APPEND TFLITE_SRCS "${TFLite230FarmhashSource_DIR}/src/farmhash.cc")
list(APPEND TFLITE_SRCS "${TFLite230FFT2DSource_DIR}/fftsg.c")
list(APPEND TFLITE_SRCS "${TFLite230FFT2DSource_DIR}/fftsg2d.c")
list(APPEND TFLITE_SRCS "${TFLite230FlatBuffersSource_DIR}/src/util.cpp")

# externals - absl
file(GLOB_RECURSE ABSL_SRCS "${TFLite230AbslSource_DIR}/absl/*.cc")
file(GLOB_RECURSE ABSL_EXCLS "${TFLite230AbslSource_DIR}/absl/*test*.cc"
     "${TFLite230AbslSource_DIR}/absl/*benchmark*.cc"
     "${TFLite230AbslSource_DIR}/absl/synchronization/*.cc"
     "${TFLite230AbslSource_DIR}/absl/debugging/*.cc"
     "${TFLite230AbslSource_DIR}/absl/hash/*.cc"
     "${TFLite230AbslSource_DIR}/absl/flags/*.cc")
list(REMOVE_ITEM ABSL_SRCS ${ABSL_EXCLS})
list(APPEND TFLITE_SRCS ${ABSL_SRCS})

# ruy lib
file(GLOB RUY_SRCS "${TFLite230RuySource_DIR}/ruy/*.cc")
file(GLOB RUY_TESTS "${TFLite230RuySource_DIR}/ruy/*test*.cc")
list(REMOVE_ITEM RUY_SRCS ${RUY_TESTS})

set(RUY_BASE ${TFLite230RuySource_DIR}/ruy)
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/benchmark.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/example.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/example_advanced.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/tune_tool.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/pmu.cc")
list(REMOVE_ITEM RUY_SRCS "${RUY_BASE}/prepare_packed_matrices.cc")

list(APPEND TFLITE_SRCS ${RUY_SRCS})

# Build with mmap? true
# caution: v2.2.0's Makefile has wrong code on this part. This is fixed on master branch.
set(BUILD_WITH_MMAP TRUE)
if(${BUILD_WITH_MMAP})
  list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/mmap_allocation_disabled.cc")
else()
  list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/mmap_allocation.cc")
endif()

# Build with nnapi? true
# caution: this nnapi delegate comes from tflite, not ours.
set(BUILD_WITH_NNAPI TRUE)
if(${BUILD_WITH_NNAPI})
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/delegates/nnapi/nnapi_delegate.cc")
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/delegates/nnapi/quant_lstm_sup.cc")
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/nnapi/nnapi_implementation.cc")
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/nnapi/nnapi_util.cc")
else()
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/delegates/nnapi/nnapi_delegate_disabled.cc")
  list(APPEND TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/nnapi/nnapi_implementation_disabled.cc")
endif()

# ios: we don't support ios
list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/minimal_logging_ios.cc")

# android
if(NOT ANDROID)
  list(REMOVE_ITEM TFLITE_SRCS "${TENSORFLOW_LITE_BASE}/minimal_logging_android.cc")
endif()

# exclude some source files
file(GLOB_RECURSE TFLITE_EXCLS "${TENSORFLOW_LITE_BASE}/*test*.cc"
     "${TENSORFLOW_LITE_BASE}/*benchmark*.cc"
     "${TENSORFLOW_LITE_BASE}/*example*.cc"
     "${TENSORFLOW_LITE_BASE}/*tool*.cc")
list(REMOVE_ITEM TFLITE_SRCS ${TFLITE_EXCLS})

# include headers
list(APPEND TFLITE_INCLUDES "${TFLite230TensorFlowSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite230EigenSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite230AbslSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite230GEMMLowpSource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite230NEON2SSESource_DIR}")
list(APPEND TFLITE_INCLUDES "${TFLite230FarmhashSource_DIR}/src")
list(APPEND TFLITE_INCLUDES "${TFLite230FlatBuffersSource_DIR}/include")
list(APPEND TFLITE_INCLUDES "${TFLite230FP16Source_DIR}/include")
list(APPEND TFLITE_INCLUDES "${TFLite230RuySource_DIR}")

add_library(tensorflow-lite-2.3.0 STATIC ${TFLITE_SRCS})
target_include_directories(tensorflow-lite-2.3.0 SYSTEM PUBLIC ${TFLITE_INCLUDES})
target_compile_definitions(tensorflow-lite-2.3.0 PUBLIC "GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -DTFLITE_WITH_RUY -DTFLITE_WITH_RUY_GEMV")
set_property(TARGET tensorflow-lite-2.3.0 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tensorflow-lite-2.3.0 eigen ${LIB_PTHREAD} dl)
if(${BUILD_WITH_NNAPI})
  target_link_libraries(tensorflow-lite-2.3.0 rt)
endif()

if(ANDROID)
  target_link_libraries(tensorflow-lite-2.3.0 log)
  target_include_directories(tensorflow-lite-2.3.0 PUBLIC "${NDK_DIR}/..")
endif()
