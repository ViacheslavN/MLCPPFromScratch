cmake_minimum_required(VERSION 3.17)

# EvaluationLib: how good a model is. Metrics for regression (MSE, RMSE, MAE, R^2),
# classification (accuracy, confusion matrix, precision, recall, F1, ROC curve and AUC,
# log-loss) and clustering (inertia, silhouette).
set(LIB_NAME EvaluationLib)

add_library(${LIB_NAME} STATIC
		Metrics/MetricsUtils.h
		Metrics/RegressionMetrics.h
		Metrics/ClassificationMetrics.h
		Metrics/ClusteringMetrics.h
		Metrics/Metrics.cpp
)
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(${LIB_NAME} PUBLIC NumericsCoreLib CommonLib)
if(MSVC)
	target_compile_options(${LIB_NAME} PRIVATE /W4 /utf-8)
else()
	target_compile_options(${LIB_NAME} PRIVATE -Wall -Wextra)
endif()

if(MLCPP_BUILD_TESTS)
	add_subdirectory(Test)
endif()
