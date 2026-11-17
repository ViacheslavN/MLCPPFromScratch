cmake_minimum_required(VERSION 3.17)

# NumericsCoreLib: vectors, matrices, tensors, linear algebra (products, decompositions),
# random numbers and basic statistics. The base of every other ML library; it depends
# only on CommonLib.
set(LIB_NAME NumericsCoreLib)

add_library(${LIB_NAME} STATIC
		Matrix.h
		Matrix.cpp
		Vector.h
		Vector.cpp
		Decompositions.h
		Decompositions.cpp
		Statistics/Statistics.h
		Statistics/Statistics.cpp
)
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(${LIB_NAME} PUBLIC CommonLib)
if(MSVC)
	target_compile_options(${LIB_NAME} PRIVATE /W4 /utf-8)
else()
	target_compile_options(${LIB_NAME} PRIVATE -Wall -Wextra)
endif()
