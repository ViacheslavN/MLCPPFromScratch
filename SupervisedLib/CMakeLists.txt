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
		LinearRegression.h
		LinearRegression.cpp
		PolynomialRegression.h
		PolynomialRegression.cpp
		LogisticRegression.h
		LogisticRegression.cpp
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
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(${LIB_NAME} PUBLIC NumericsCoreLib CommonLib)
if(MSVC)
	target_compile_options(${LIB_NAME} PRIVATE /W4 /utf-8)
else()
	target_compile_options(${LIB_NAME} PRIVATE -Wall -Wextra)
endif()
