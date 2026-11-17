#include "TestCommon.h"

using namespace ML::numerics;
using namespace ML::evaluation;
using namespace test_utils;

TEMPLATE_TEST_CASE("Regression metrics of known values", "[metrics][regression]", int, float, double)
{
	// errors: 1, -1, 2, 0
	const TVector<TestType> yTrue = MakeVector<TestType>({3, 5, 2, 7});
	const TVector<TestType> yPredicted = MakeVector<TestType>({2, 6, 0, 7});

	REQUIRE(CRegressionMetrics::MSE(yTrue, yPredicted) == Near(6.0 / 4));
	REQUIRE(CRegressionMetrics::RMSE(yTrue, yPredicted) == Near(std::sqrt(1.5)));
	REQUIRE(CRegressionMetrics::MAE(yTrue, yPredicted) == Near(4.0 / 4));

	// mean of yTrue is 4.25; total sum of squares 14.75
	REQUIRE(CRegressionMetrics::R2(yTrue, yPredicted) == Near(1 - 6.0 / 14.75));

	// the result is a double whatever the element type
	STATIC_REQUIRE(std::is_same<decltype(CRegressionMetrics::MSE(yTrue, yPredicted)), double>::value);
}

TEMPLATE_TEST_CASE("Regression metrics: perfect, mean and bad predictions", "[metrics][regression]", float, double)
{
	const TVector<TestType> yTrue = MakeVector<TestType>({1, 2, 3, 4, 5});

	// perfect
	REQUIRE(CRegressionMetrics::MSE(yTrue, yTrue) == 0);
	REQUIRE(CRegressionMetrics::RMSE(yTrue, yTrue) == 0);
	REQUIRE(CRegressionMetrics::MAE(yTrue, yTrue) == 0);
	REQUIRE(CRegressionMetrics::R2(yTrue, yTrue) == 1);

	// always the mean: R^2 = 0, MSE = the population variance
	const TVector<TestType> mean(5, TestType(3));
	REQUIRE(CRegressionMetrics::R2(yTrue, mean) == Near(0));
	REQUIRE(CRegressionMetrics::MSE(yTrue, mean) == Near(2));

	// worse than the mean: negative R^2
	const TVector<TestType> reversed = MakeVector<TestType>({5, 4, 3, 2, 1});
	REQUIRE(CRegressionMetrics::R2(yTrue, reversed) == Near(-3));

	// RMSE punishes one large error more than MAE does
	const TVector<TestType> oneBigError = MakeVector<TestType>({1, 2, 3, 4, 15});
	REQUIRE(CRegressionMetrics::MAE(yTrue, oneBigError) == Near(2));
	REQUIRE(CRegressionMetrics::RMSE(yTrue, oneBigError) == Near(std::sqrt(20.0)));

	// constant targets: R^2 is 0 / 0
	const TVector<TestType> constant(3, TestType(2));
	REQUIRE(CRegressionMetrics::R2(constant, constant) == 1);
	REQUIRE(CRegressionMetrics::R2(constant, MakeVector<TestType>({2, 2, 3})) == 0);
}

TEMPLATE_TEST_CASE("Regression metrics reject wrong sizes", "[metrics][regression]", int, float, double)
{
	const TVector<TestType> three(3);
	const TVector<TestType> two(2);
	const TVector<TestType> empty;

	REQUIRE_THROWS_AS(CRegressionMetrics::MSE(three, two), CExcBase);
	REQUIRE_THROWS_AS(CRegressionMetrics::RMSE(three, two), CExcBase);
	REQUIRE_THROWS_AS(CRegressionMetrics::MAE(three, two), CExcBase);
	REQUIRE_THROWS_AS(CRegressionMetrics::R2(three, two), CExcBase);

	REQUIRE_THROWS_AS(CRegressionMetrics::MSE(empty, empty), CExcBase);
	REQUIRE_THROWS_AS(CRegressionMetrics::MAE(empty, empty), CExcBase);
	REQUIRE_THROWS_AS(CRegressionMetrics::R2(empty, empty), CExcBase);
}
