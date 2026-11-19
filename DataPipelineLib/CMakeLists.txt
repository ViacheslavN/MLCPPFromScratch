cmake_minimum_required(VERSION 3.17)

# DataPipelineLib: CSV reader, dataset type, preprocessing (missing values, encoding,
# scaling), train/validation/test split, k-fold and mini-batch iteration.
set(LIB_NAME DataPipelineLib)

add_library(${LIB_NAME} STATIC
		ICSVFile.h
		CSVConverter.h
		CSVConverter.cpp
		CSVFile.h
		CSVFile.cpp
		CSVPrinter.h
		CSVPrinter.cpp
		Splitting/Splitting.h
		Splitting/Splitting.cpp
		Preprocessing/StandardScaler.h
		Preprocessing/StandardScaler.cpp
)
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(${LIB_NAME} PUBLIC NumericsCoreLib CommonLib)
if(MSVC)
	target_compile_options(${LIB_NAME} PRIVATE /W4 /utf-8)
else()
	target_compile_options(${LIB_NAME} PRIVATE -Wall -Wextra)
endif()
