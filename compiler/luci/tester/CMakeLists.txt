nnas_include(TargetRequire)

unset(REQUIRED_TARGETS)
list(APPEND REQUIRED_TARGETS safemain)
TargetRequire_Return(${REQUIRED_TARGETS})

set(SRCS_READ_TESTER
      src/ReadTester.cpp
   )

add_executable(luci_readtester "${SRCS_READ_TESTER}")
target_link_libraries(luci_readtester PRIVATE luci_import)
target_link_libraries(luci_readtester PRIVATE luci_service)
target_link_libraries(luci_readtester PRIVATE luci_pass)
target_link_libraries(luci_readtester PRIVATE foder)
target_link_libraries(luci_readtester PRIVATE oops)
target_link_libraries(luci_readtester PRIVATE safemain)

set(SRCS_WRITE_TESTER
      src/WriteTester.cpp
   )

add_executable(luci_writetester "${SRCS_WRITE_TESTER}")
target_link_libraries(luci_writetester PRIVATE luci_import)
target_link_libraries(luci_writetester PRIVATE luci_service)
target_link_libraries(luci_writetester PRIVATE luci_pass)
target_link_libraries(luci_writetester PRIVATE luci_export)
target_link_libraries(luci_writetester PRIVATE foder)
target_link_libraries(luci_writetester PRIVATE oops)
target_link_libraries(luci_writetester PRIVATE safemain)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(luci_readtester_test src/ReadTester.test.cpp src/ReadTester.cpp)
target_link_libraries(luci_readtester_test luci_import)
target_link_libraries(luci_readtester_test luci_service)
target_link_libraries(luci_readtester_test luci_pass)
target_link_libraries(luci_readtester_test foder)

GTest_AddTest(luci_writetester_test src/WriteTester.test.cpp src/WriteTester.cpp)
target_link_libraries(luci_writetester_test luci_import)
target_link_libraries(luci_writetester_test luci_service)
target_link_libraries(luci_writetester_test luci_pass)
target_link_libraries(luci_writetester_test luci_export)
target_link_libraries(luci_writetester_test foder)
