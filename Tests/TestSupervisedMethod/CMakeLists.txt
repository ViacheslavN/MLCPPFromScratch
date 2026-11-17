set(BIN_NAME TestSupervisedMethod)

set(SOURCE_FILES
        main.cpp
        LinearRegressionTest.cpp
)
if (WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
add_executable(${BIN_NAME} ${SOURCE_FILES})
target_compile_definitions(${BIN_NAME} PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../Data")
target_link_libraries(${BIN_NAME} PUBLIC CommonLib)
target_link_libraries(${BIN_NAME} PUBLIC NumericsCoreLib)
target_link_libraries(${BIN_NAME} PUBLIC SupervisedLib)
target_link_libraries(${BIN_NAME} PUBLIC DataPipelineLib)
target_link_libraries(${BIN_NAME} PUBLIC EvaluationLib)