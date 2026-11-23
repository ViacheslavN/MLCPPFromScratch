#pragma once

#include "SupervisedLib/DecisionTree.h"
#include "SupervisedLib/LinearRegression/LinearRegression.h"
#include "SupervisedLib/LinearRegression/RidgeRegression.h"
#include "SupervisedLib/LinearRegression/LassoRegression.h"
#include "SupervisedLib/KNearestNeighbors.h"


template<class TModel, class TMatrix, class TVector>
class TRegressionTestSimple {
    public:
    TRegressionTestSimple(){}
    ~TRegressionTestSimple(){}

    void Test(const std::string& fileCVSPath);

};


typedef TRegressionTestSimple< ML::supervised::CLinearRegression,  ML::numerics::CMatrix ,  ML::numerics::CVector> TRegressionTestSimpleLinearRegression;
typedef TRegressionTestSimple< ML::supervised::CKNNRegressor,  ML::numerics::CMatrix ,  ML::numerics::CVector> TRegressionTestSimpleKNNRegressor;
typedef TRegressionTestSimple< ML::supervised::CDecisionTreeRegressor,  ML::numerics::CMatrix ,  ML::numerics::CVector> TRegressionTestSimpleDecisionTreeRegressor;
typedef TRegressionTestSimple< ML::supervised::CRidgeRegression,  ML::numerics::CMatrix ,  ML::numerics::CVector> TRegressionTestSimpleRidgeRegression;
typedef TRegressionTestSimple< ML::supervised::CLassoRegression,  ML::numerics::CMatrix ,  ML::numerics::CVector> TRegressionTestSimpleLassoRegression;



