if(BUILD_TENSORFLOW_LITE_2_3_0)
  macro(return_unless VAR)
  if(NOT ${VAR})
    message("${VAR} NOT TRUE")
    set(TensorFlowLite_2_3_0_FOUND PARENT_SCOPE)
    return()
  endif(NOT ${VAR})
  endmacro(return_unless)

  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # Below urls come from https://github.com/tensorflow/tensorflow/blob/v2.3.0-rc0/tensorflow/lite/tools/make/Makefile

  set(absl_url "https://github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz")
  ExternalSource_Download("tflite230_Absl" ${absl_url})
  set(TFLite230AbslSource_DIR "${tflite230_Absl_SOURCE_DIR}")
  if (NOT TFLite230AbslSource_DIR STREQUAL "")
    set(TFLite230AbslSource_FOUND TRUE)
  endif()
  return_unless(TFLite230AbslSource_FOUND)

  set(eigen_url "https://gitlab.com/libeigen/eigen/-/archive/52a2fbbb008a47c5e3fb8ac1c65c2feecb0c511c/eigen-52a2fbbb008a47c5e3fb8ac1c65c2feecb0c511c.tar.gz")
  ExternalSource_Download("tflite230_Eigen" ${eigen_url})
  set(TFLite230EigenSource_DIR "${tflite230_Eigen_SOURCE_DIR}")
  if (NOT TFLite230EigenSource_DIR STREQUAL "")
    set(TFLite230EigenSource_FOUND TRUE)
  endif()
  return_unless(TFLite230EigenSource_FOUND)

  set(farmhash_url "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz")
  ExternalSource_Download("tflite230_Farmhash" ${farmhash_url})
  set(TFLite230FarmhashSource_DIR "${tflite230_Farmhash_SOURCE_DIR}")
  if (NOT TFLite230FarmhashSource_DIR STREQUAL "")
    set(TFLite230FarmhashSource_FOUND TRUE)
  endif()
  return_unless(TFLite230FarmhashSource_FOUND)

  set(fft2d_url "https://storage.googleapis.com/mirror.tensorflow.org/www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz")
  ExternalSource_Download("tflite230_FFT2D" ${fft2d_url})
  set(TFLite230FFT2DSource_DIR "${tflite230_FFT2D_SOURCE_DIR}")
  if (NOT TFLite230FFT2DSource_DIR STREQUAL "")
    set(TFLite230FFT2DSource_FOUND TRUE)
  endif()
  return_unless(TFLite230FFT2DSource_FOUND)

  set(flatbuffers_url "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v1.12.0.tar.gz")
  ExternalSource_Download("tflite230_FlatBuffers" ${flatbuffers_url})
  set(TFLite230FlatBuffersSource_DIR "${tflite230_FlatBuffers_SOURCE_DIR}")
  if (NOT TFLite230FlatBuffersSource_DIR STREQUAL "")
    set(TFLite230FlatBuffersSource_FOUND TRUE)
  endif()
  return_unless(TFLite230FlatBuffersSource_FOUND)

  set(fp16_url "https://github.com/Maratyszcza/FP16/archive/febbb1c163726b5db24bed55cc9dc42529068997.zip")
  ExternalSource_Download("tflite230_FP16" ${fp16_url})
  set(TFLite230FP16Source_DIR "${tflite230_FP16_SOURCE_DIR}")
  if (NOT TFLite230FP16Source_DIR STREQUAL "")
    set(TFLite230FP16Source_FOUND TRUE)
  endif()
  return_unless(TFLite230FP16Source_FOUND)

  set(gemmlowp_url "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip")
  ExternalSource_Download("tflite230_GEMMLowp" ${gemmlowp_url})
  set(TFLite230GEMMLowpSource_DIR "${tflite230_GEMMLowp_SOURCE_DIR}")
  if (NOT TFLite230GEMMLowpSource_DIR STREQUAL "")
    set(TFLite230GEMMLowpSource_FOUND TRUE)
  endif()
  return_unless(TFLite230GEMMLowpSource_FOUND)

  set(neon2sse_url "https://github.com/intel/ARM_NEON_2_x86_SSE/archive/master.zip")
  ExternalSource_Download("tflite230_NEON2SSE" ${neon2sse_url})
  set(TFLite230NEON2SSESource_DIR "${tflite230_NEON2SSE_SOURCE_DIR}")
  if (NOT TFLite230NEON2SSESource_DIR STREQUAL "")
    set(TFLite230NEON2SSESource_FOUND TRUE)
  endif()
  return_unless(TFLite230NEON2SSESource_FOUND)

  set(ruy_url "https://github.com/google/ruy/archive/34ea9f4993955fa1ff4eb58e504421806b7f2e8f.zip")
  ExternalSource_Download("tflite230_Ruy" ${ruy_url})
  set(TFLite230RuySource_DIR "${tflite230_Ruy_SOURCE_DIR}")
  if (NOT TFLite230RuySource_DIR STREQUAL "")
    set(TFLite230RuySource_FOUND TRUE)
  endif()
  return_unless(TFLite230RuySource_FOUND)

  set(tensorflow_url "https://github.com/tensorflow/tensorflow/archive/v2.3.0-rc0.tar.gz")
  ExternalSource_Download("tflite230_TensorFlow" ${tensorflow_url})
  set(TFLite230TensorFlowSource_DIR "${tflite230_TensorFlow_SOURCE_DIR}")
  if (NOT TFLite230TensorFlowSource_DIR STREQUAL "")
    set(TFLite230TensorFlowSource_FOUND TRUE)
  endif()
  return_unless(TFLite230TensorFlowSource_FOUND)

  nnas_include(ExternalProjectTools)
  add_extdirectory("${CMAKE_CURRENT_LIST_DIR}/TensorFlowLite-2.3.0" tflite-2.3.0)

  set(TensorFlowLite_2_3_0_FOUND TRUE)
  return()
endif()
