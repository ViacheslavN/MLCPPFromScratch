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
#   NumericsCore  - vectors, matrices, linear algebra, random numbers   NumericsCoreLib
#   DataPipeline  - CSV, preprocessing, splitting, batching             DataPipelineLib
#   Optimization  - gradient descent and its variants                   OptimizationLib
#   Evaluation    - losses, metrics, cross-validation                   EvaluationLib
#   SupervisedLib - regression and classification models               SupervisedLib
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

add_subdirectory(CommonLib)
add_subdirectory(NumericsCore)
add_subdirectory(DataPipeline)
add_subdirectory(Optimization)
add_subdirectory(Evaluation)
add_subdirectory(SupervisedLib)
