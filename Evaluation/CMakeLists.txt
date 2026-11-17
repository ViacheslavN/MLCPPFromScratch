cmake_minimum_required(VERSION 3.17)

# EvaluationLib: loss functions, regression, classification and clustering metrics,
# cross-validation. Header-only for now.
add_library(EvaluationLib INTERFACE)
target_include_directories(EvaluationLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(EvaluationLib INTERFACE NumericsCoreLib CommonLib)
