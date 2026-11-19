#include "stdafx.h"
#include "RegressionTest.h"
#include "DataPipelineLib/CSVFile.h"
#include "DataPipelineLib/CSVPrinter.h"
#include "exception/exc_base.h"
#include "log/logger.h"

#include "DataPipelineLib/Splitting/Splitting.h"
#include "EvaluationLib/Metrics/RegressionMetrics.h"

template<class TModel, class TMatrix, class TVector>
void TRegressionTestSimple<TModel, TMatrix, TVector>::Test(const std::string& fileCVSPath)
{
   try
   {

        Log.Info("Loading data from CSV files... ");
        ML::data::ICSVFilePtr ptrFile =  ML::data::CCSVFile::Open(fileCVSPath);
        ML::data::CCSVPrinter::Head(ptrFile, 3);

        TMatrix X;
        TVector Y;

        Log.Info("Creating matrix and vector... ");
        ptrFile->CreateMatrixDrop({"Sales"}, X);
        ptrFile->CreateVector("Sales", Y);

        Log.Info("Features:");
        ML::data::CCSVPrinter::Head(X);

        Log.Info("Splitting:");
        ML::data::TSplit<double, double> splitData =  ML::data::CSplitting::Split(X,Y);

        Log.Info("Train data:");
        ML::data::CCSVPrinter::Head(splitData.train.x);

        Log.Info("Test data:");
        ML::data::CCSVPrinter::Head(splitData.test.x);

        Log.Info("Fitting:");

        TModel model;
        model.Fit(splitData.train.x, splitData.train.y);

        Log.Info("Predicting:");
        ML::numerics::CVector y_pred = model.Predict(splitData.test.x);

        double MAE =  ML::evaluation::CRegressionMetrics::MAE(splitData.test.y,y_pred);
        double MSE =  ML::evaluation::CRegressionMetrics::MSE(splitData.test.y,y_pred);
        double RMSE =  ML::evaluation::CRegressionMetrics::RMSE(splitData.test.y,y_pred);
        double R2 =  ML::evaluation::CRegressionMetrics::R2(splitData.test.y,y_pred);

        Log.InfoT("MAE: {0}, MSE: {1}, RMSE: {2}, R2: {3}",MAE,MSE,RMSE,R2);
        //Log.Info("Weights:");
       // const TVector& w = model.GetWeights();
       //ML::data::CCSVPrinter::Head(w, w.Size());


    }catch (std::exception& e) {
        CommonLib::CExcBase::RegenExc("CLinearRegressionTest", e);
    }
}

// The template is defined in this file, so the versions that are used elsewhere must be
// instantiated here; without this line the linker does not find Test().
template class TRegressionTestSimple<ML::supervised::CLinearRegression, ML::numerics::CMatrix, ML::numerics::CVector>;
template class TRegressionTestSimple<ML::supervised::CKNNRegressor, ML::numerics::CMatrix, ML::numerics::CVector>;
template class TRegressionTestSimple<ML::supervised::CDecisionTreeRegressor, ML::numerics::CMatrix, ML::numerics::CVector>;
template class TRegressionTestSimple<ML::supervised::CRidgeRegression, ML::numerics::CMatrix, ML::numerics::CVector>;
template class TRegressionTestSimple<ML::supervised::CLassoRegression, ML::numerics::CMatrix, ML::numerics::CVector>;


