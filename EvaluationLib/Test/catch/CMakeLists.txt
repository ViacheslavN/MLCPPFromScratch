cmake_minimum_required(VERSION 3.17)

# Catch2 unit tests of EvaluationLib: regression, classification and clustering metrics.
include(${PROJECT_SOURCE_DIR}/cmake/Catch2.cmake)
if(NOT MLCPP_CATCH2_FOUND)
	message(STATUS "Catch2 v3 not found - EvaluationLib Catch tests are skipped "
			"(install it, or configure with -DMLCPP_FETCH_CATCH2=ON to download it)")
	return()
endif()

set(BIN_NAME EvaluationCatchTests)

add_executable(${BIN_NAME}
		TestCommon.h
		TestRegressionMetrics.cpp
		TestClassificationMetrics.cpp
		TestClusteringMetrics.cpp
)

# EvaluationLib brings the include path of the repo root: "EvaluationLib/xxx.h".
target_link_libraries(${BIN_NAME} PRIVATE
		EvaluationLib
		Catch2::Catch2WithMain
)
if(MSVC)
	target_compile_options(${BIN_NAME} PRIVATE /utf-8)
endif()

# Register every TEST_CASE with CTest
include(CTest)
include(Catch)
catch_discover_tests(${BIN_NAME})
