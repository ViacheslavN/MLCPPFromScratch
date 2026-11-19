#include "stdafx.h"

#include "log/logger.h"
#include "../../CommonLib/log/ConsoleLogWriter.h"
#include "RegressionTest.h"

int main() {

    std::shared_ptr<CommonLib::IlogWriter> consoleLogger = std::make_shared<CommonLib::CConsoleLogWriter>();
    Log.InnitLog(std::shared_ptr<CommonLib::IlogLineFormat>(), consoleLogger);
    try {


        Log.Info("Linear Regression Test:");

        TRegressionTestSimpleLinearRegression linearRegressionTest;
        linearRegressionTest.Test(TEST_DATA_DIR"/Advertising.csv");


        Log.Info("KNN Regression Test:");
        TRegressionTestSimpleKNNRegressor  KNNRegressor;
        KNNRegressor.Test(TEST_DATA_DIR"/Advertising.csv");

        Log.Info("Decision Regression Test:");
        TRegressionTestSimpleDecisionTreeRegressor  DecisionTreeRegressor;
        DecisionTreeRegressor.Test(TEST_DATA_DIR"/Advertising.csv");

        Log.Info("Rindge Regression Test:");
        TRegressionTestSimpleRidgeRegression ridgeRegression;
        ridgeRegression.Test(TEST_DATA_DIR"/Advertising.csv");

        Log.Info("Lasso Regression Test:");
        TRegressionTestSimpleLassoRegression lassoRegression;
        lassoRegression.Test(TEST_DATA_DIR"/Advertising.csv");


    }
    catch (std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
