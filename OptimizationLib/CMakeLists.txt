cmake_minimum_required(VERSION 3.17)

# OptimizationLib: gradient descent, SGD and mini-batch SGD, momentum, AdaGrad, RMSProp,
# Adam, learning-rate schedules and early stopping. Header-only for now.
add_library(OptimizationLib INTERFACE)
target_include_directories(OptimizationLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(OptimizationLib INTERFACE NumericsCoreLib CommonLib)
