#include "stdafx.h"
#include "LinearRegressionTest.h"
#include "log/logger.h"
#include "../../CommonLib/log/ConsoleLogWriter.h"

int main() {

    std::shared_ptr<CommonLib::IlogWriter> consoleLogger = std::make_shared<CommonLib::CConsoleLogWriter>();
    Log.InnitLog(std::shared_ptr<CommonLib::IlogLineFormat>(), consoleLogger);
    try {

        CLinearRegressionTest test;
        test.Test();

    }
    catch (std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
