function(_Fp16_Build)
  nnas_find_package(Fp16Source QUIET)

  # NOTE This line prevents multiple definitions of target
  if(TARGET fp16)
    set(Fp16Source_DIR ${PsimdSource_DIR} PARENT_SCOPE)
    set(Fp16_FOUND TRUE PARENT_SCOPE)
    return()
  endif(TARGET fp16)

  if(NOT Fp16Source_FOUND)
    message(STATUS "FP16: Source not found")
    set(Fp16_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT Fp16Source_FOUND)

  set(FP16_BUILD_TESTS OFF CACHE BOOL "Build FP16 unit tests")
  set(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "Build FP16 micro-benchmarks")
  
  add_extdirectory("${Fp16Source_DIR}" FP16 EXCLUDE_FROM_ALL)
  set(Fp16Source_DIR ${Fp16Source_DIR} PARENT_SCOPE)
  set(Fp16_FOUND TRUE PARENT_SCOPE)
endfunction(_Fp16_Build)

if(BUILD_FP16)
  _Fp16_Build()
else()
  set(Fp16_FOUND FALSE)
endif()
