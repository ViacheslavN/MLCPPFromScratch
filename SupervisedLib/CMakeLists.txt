cmake_minimum_required(VERSION 3.17)

# SupervisedLib: supervised learning models behind two interfaces, IRegressor and
# IClassifier (Fit, Predict, Score): linear, polynomial, ridge and lasso regression, logistic
# regression, k nearest neighbors, Gaussian Naive Bayes, decision trees, random forests,
# gradient boosting and a support vector machine.
set(LIB_NAME SupervisedLib)

add_library(${LIB_NAME} STATIC
		Model.h
		Model.cpp
		ModelUtils.h
		LinearRegression/LinearRegression.h
		LinearRegression/LinearRegression.cpp
		LinearRegression/RidgeRegression.h
		LinearRegression/RidgeRegression.cpp
		LinearRegression/LassoRegression.h
		LinearRegression/LassoRegression.cpp
		LinearRegression/CenteredData.h
		LinearRegression/PolynomialRegression.h
		LinearRegression/PolynomialRegression.cpp
		LogisticRegression/LogisticRegression.h
		LogisticRegression/LogisticRegression.cpp
		KNearestNeighbors.h
		KNearestNeighbors.cpp
		NaiveBayes.h
		NaiveBayes.cpp
		DecisionTree.h
		DecisionTree.cpp
		RandomForest.h
		RandomForest.cpp
		GradientBoosting.h
		GradientBoosting.cpp
		SVM.h
		SVM.cpp
)
if (WIN32)
	add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# DataPipelineLib: the polynomial regression uses its polynomial features.
target_link_libraries(${LIB_NAME} PUBLIC DataPipelineLib NumericsCoreLib CommonLib)
if(MSVC)
	target_compile_options(${LIB_NAME} PRIVATE /W4 /utf-8)
else()
	target_compile_options(${LIB_NAME} PRIVATE -Wall -Wextra)
endif()
