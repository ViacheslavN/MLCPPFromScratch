cmake_minimum_required(VERSION 3.17)

# vcpkg: static libraries built with the static CRT (/MT), the same as this project
# (CMAKE_MSVC_RUNTIME_LIBRARY below), so OpenSSL links into the exe and no DLLs have
# to be shipped. Read by the vcpkg toolchain, so it must come before project().
# Override with -DVCPKG_TARGET_TRIPLET=... if needed.
if(CMAKE_HOST_WIN32 AND NOT DEFINED VCPKG_TARGET_TRIPLET)
	set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "vcpkg triplet")
endif()

project(MLCPPFromScratch CXX C)

# MLCPPFromScratch: machine learning algorithms implemented from scratch in C++,
# for education (supervised learning, unsupervised learning, deep learning).
#   CommonLib     - shared utilities (copied from LanguageTrainer)
# The folders are named after the library targets they build:
#   NumericsCoreLib  - vectors, matrices, decompositions, statistics
#   DataPipelineLib  - CSV, preprocessing, splitting, batching
#   OptimizationLib  - gradient descent and its variants
#   EvaluationLib    - losses, metrics, cross-validation
#   SupervisedLib    - regression and classification models
# The other model libraries (unsupervised, deep learning) are added below as they are
# written, one add_subdirectory() per library.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static CRT on MSVC (/MT, /MTd), the same as CommonLib and its Catch2 tests expect.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(MSVC)
	add_compile_options(/MP)
	# No global NOMINMAX or warning level: CommonLib relies on the <windows.h>
	# min/max macros. Strict options are set per ML target instead.
endif()

# CommonLib reads this option for its own Catch2 tests.
option(GRAPHENGINE_BUILD_TESTS "Build CommonLib tests" OFF)

# Tests of the ML libraries. On by default; -DMLCPP_BUILD_TESTS=OFF to skip.
option(MLCPP_BUILD_TESTS "Build MLCPPFromScratch tests" ON)
if(MLCPP_BUILD_TESTS)
	# Lets 'ctest' run from the build folder find the tests of all the libraries.
	enable_testing()
endif()

add_subdirectory(CommonLib)
add_subdirectory(NumericsCoreLib)
add_subdirectory(DataPipelineLib)
add_subdirectory(OptimizationLib)
add_subdirectory(EvaluationLib)
add_subdirectory(SupervisedLib)
add_subdirectory(Tests)