#include "stdafx.h"
#include "LinearRegressionTest.h"

#include <exception>

#include "DataPipelineLib/CSVFile.h"
#include "DataPipelineLib/CSVPrinter.h"
#include "exception/exc_base.h"
#include "log/logger.h"
#include "DataPipelineLib/Splitting/Splitting.h"
#include "EvaluationLib/Metrics/RegressionMetrics.h"


void CLinearRegressionTest::Test() {

    try {

        Log.Info("Loading data from CSV files... ");
        ML::data::ICSVFilePtr ptrFile =  ML::data::CCSVFile::Open(TEST_DATA_DIR"/Advertising.csv");
        ML::data::CCSVPrinter::Head(ptrFile, 3);

        ML::numerics::CMatrix X;
        ML::numerics::CVector Y;

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

        ML::supervised::CLinearRegression model;
        model.Fit(splitData.train.x, splitData.train.y);

        Log.Info("Predicting:");
        ML::numerics::CVector y_pred = model.Predict(splitData.test.x);

        double MAE =  ML::evaluation::CRegressionMetrics::MAE(splitData.test.y,y_pred);
        double MSE =  ML::evaluation::CRegressionMetrics::MSE(splitData.test.y,y_pred);
        double RMSE =  ML::evaluation::CRegressionMetrics::RMSE(splitData.test.y,y_pred);
        double R2 =  ML::evaluation::CRegressionMetrics::R2(splitData.test.y,y_pred);

        Log.InfoT("MAE: {0}, MSE: {1}, RMSE: {2}, R2: {3}",MAE,MSE,RMSE,R2);

    }catch (std::exception& e) {
        CommonLib::CExcBase::RegenExc("CLinearRegressionTest", e);
    }



}
