#include "TestCommon.h"

#include "NumericsCore/Statistics/Statistics.h"

using namespace ML::numerics;
using namespace test_utils;

// ---------------------------------------------------------------------------
// Mean and variance
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Statistics mean", "[statistics]", float, double)
{
	typedef TStatistics<TestType> TStat;
	typedef TMatrix<TestType> TMat;

	REQUIRE(TStat::Mean(MakeVector<TestType>({2, 4, 4, 4, 5, 5, 7, 9})) == 5);
	REQUIRE_THROWS_AS(TStat::Mean(TVector<TestType>()), CExcBase);

	const TMat matrix = MakeMatrix<TestType>(3, 2, {1, 2, 3, 4, 5, 9});
	REQUIRE(MaxDiff(TStat::Mean(matrix, TMat::axisRows), {3, 5}) == 0);
	REQUIRE(MaxDiff(TStat::Mean(matrix, TMat::axisCols), {1.5, 3.5, 7}) == 0);
}

TEMPLATE_TEST_CASE("Statistics variance", "[statistics]", float, double)
{
	typedef TStatistics<TestType> TStat;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();

	// mean 5, squared deviations sum to 32
	const TVector<TestType> data = MakeVector<TestType>({2, 4, 4, 4, 5, 5, 7, 9});
	REQUIRE(TStat::Variance(data, vtPopulation) == Catch::Approx(4).margin(tolerance));
	REQUIRE(TStat::Variance(data, vtSample) == Catch::Approx(32.0 / 7).margin(tolerance));
	REQUIRE(TStat::Variance(data) == TStat::Variance(data, vtSample));		// the default is the sample variance

	REQUIRE(TStat::Variance(MakeVector<TestType>({3, 3, 3, 3})) == 0);
	REQUIRE(TStat::Variance(MakeVector<TestType>({42}), vtPopulation) == 0);

	// too few values: 2 for a sample, 1 for a population
	REQUIRE_THROWS_AS(TStat::Variance(MakeVector<TestType>({42})), CExcBase);
	REQUIRE_THROWS_AS(TStat::Variance(TVector<TestType>(), vtPopulation), CExcBase);

	const TMat matrix = MakeMatrix<TestType>(3, 2, {1, 2, 3, 4, 5, 9});
	REQUIRE(MaxDiff(TStat::Variance(matrix, TMat::axisRows), {4, 13}) < tolerance);
	REQUIRE(MaxDiff(TStat::Variance(matrix, TMat::axisCols), {0.5, 0.5, 8}) < tolerance);
	REQUIRE(MaxDiff(TStat::Variance(matrix, TMat::axisRows, vtPopulation), {8.0 / 3, 26.0 / 3}) < tolerance);

	// one row: no sample variance down the rows, but there is one across the columns
	const TMat row = MakeMatrix<TestType>(1, 3, {1, 2, 3});
	REQUIRE_THROWS_AS(TStat::Variance(row, TMat::axisRows), CExcBase);
	REQUIRE(MaxDiff(TStat::Variance(row, TMat::axisCols), {1}) < tolerance);
}

TEST_CASE("Statistics variance keeps its precision next to a large offset", "[statistics]")
{
	// The one-pass formula mean(x^2) - mean(x)^2 returns garbage here; the values are
	// 1, 2, 3 and 4, 7, 13, 16 shifted by a large constant.
	CVector shifted(3);
	shifted[0] = 1e9 + 1;
	shifted[1] = 1e9 + 2;
	shifted[2] = 1e9 + 3;
	REQUIRE(CStatistics::Variance(shifted) == 1.0);

	CVector shiftedFar(4);
	shiftedFar[0] = 1e15 + 4;
	shiftedFar[1] = 1e15 + 7;
	shiftedFar[2] = 1e15 + 13;
	shiftedFar[3] = 1e15 + 16;
	REQUIRE(CStatistics::Variance(shiftedFar) == 30.0);
}

// ---------------------------------------------------------------------------
// Covariance
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Statistics covariance matrix", "[statistics]", float, double)
{
	typedef TStatistics<TestType> TStat;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();

	// column means 3 and 5; deviations (-2,-3) (0,-1) (2,4)
	const TMat matrix = MakeMatrix<TestType>(3, 2, {1, 2, 3, 4, 5, 9});
	const TMat covariance = TStat::Covariance(matrix);
	REQUIRE(covariance.Rows() == 2);
	REQUIRE(covariance.Cols() == 2);
	REQUIRE(MaxDiff(covariance, 2, 2, {4, 7, 7, 13}) < tolerance);
	REQUIRE(MaxDiff(TStat::Covariance(matrix, vtPopulation), 2, 2, {8.0 / 3, 14.0 / 3, 14.0 / 3, 26.0 / 3}) < tolerance);

	// three variables: against the definition, element by element
	const TMat data = MakeMatrix<TestType>(5, 3, {4.0, 2.0, 0.60, 4.2, 2.1, 0.59, 3.9, 2.0, 0.58, 4.3, 2.1, 0.62, 4.1, 2.2, 0.63});
	const TMat cov = TStat::Covariance(data);
	const TVector<TestType> variances = TStat::Variance(data, TMat::axisRows);
	const TVector<TestType> means = TStat::Mean(data, TMat::axisRows);
	for (size_t i = 0; i < 3; ++i)
	{
		REQUIRE(double(cov(i, i)) == Catch::Approx(double(variances[i])).margin(tolerance));	// the diagonal: the variances

		for (size_t j = 0; j < 3; ++j)
		{
			REQUIRE(cov(i, j) == cov(j, i));	// exactly symmetric

			double sum = 0;
			for (size_t k = 0; k < 5; ++k)
				sum += (double(data(k, i)) - double(means[i])) * (double(data(k, j)) - double(means[j]));

			REQUIRE(double(cov(i, j)) == Catch::Approx(sum / 4).margin(tolerance));
		}
	}

	// the second column is 2 * first + 1: cov(x, y) = 2 * var(x), var(y) = 4 * var(x)
	const TMat linear = TStat::Covariance(MakeMatrix<TestType>(4, 2, {1, 3, 2, 5, 4, 9, 7, 15}));
	REQUIRE(double(linear(0, 1)) == Catch::Approx(2 * double(linear(0, 0))).margin(100 * tolerance));
	REQUIRE(double(linear(1, 1)) == Catch::Approx(4 * double(linear(0, 0))).margin(100 * tolerance));

	REQUIRE_THROWS_AS(TStat::Covariance(MakeMatrix<TestType>(1, 2, {1, 2})), CExcBase);		// one sample
	REQUIRE(MaxDiff(TStat::Covariance(MakeMatrix<TestType>(1, 2, {1, 2}), vtPopulation), TMat(2, 2)) == 0);
}

// ---------------------------------------------------------------------------
// Log-sum-exp and softmax
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Statistics log-sum-exp", "[statistics]", float, double)
{
	typedef TStatistics<TestType> TStat;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	const double ln2 = std::log(2.0);
	const double ln3 = std::log(3.0);
	const TestType infinity = std::numeric_limits<TestType>::infinity();

	REQUIRE(TStat::LogSumExp(MakeVector<TestType>({0, 0, 0})) == Catch::Approx(ln3).margin(tolerance));
	REQUIRE(TStat::LogSumExp(MakeVector<TestType>({1, 2, 3})) == Catch::Approx(std::log(std::exp(1.0) + std::exp(2.0) + std::exp(3.0))).margin(tolerance));
	REQUIRE(TStat::LogSumExp(MakeVector<TestType>({5})) == Catch::Approx(5).margin(tolerance));

	// exp(1000) overflows and exp(-1000) underflows; the result is still exact
	REQUIRE(TStat::LogSumExp(MakeVector<TestType>({1000, 1000})) == Catch::Approx(1000 + ln2).margin(1000 * tolerance));
	REQUIRE(TStat::LogSumExp(MakeVector<TestType>({-1000, -1000})) == Catch::Approx(-1000 + ln2).margin(1000 * tolerance));
	REQUIRE(TStat::LogSumExp(MakeVector<TestType>({-2000, 0})) == Catch::Approx(0).margin(tolerance));

	REQUIRE(TStat::LogSumExp(TVector<TestType>()) == -infinity);		// log(0)
	REQUIRE(TStat::LogSumExp(TVector<TestType>(2, -infinity)) == -infinity);
	REQUIRE(TStat::LogSumExp(MakeVector<TestType>({1, double(infinity)})) == infinity);
	REQUIRE(TStat::LogSumExp(MakeVector<TestType>({-double(infinity), 3})) == Catch::Approx(3).margin(tolerance));

	const TMat matrix = MakeMatrix<TestType>(2, 3, {1000, 1000, 1000, -1, 0, 1});
	REQUIRE(MaxDiff(TStat::LogSumExp(matrix, TMat::axisCols), {1000 + ln3, std::log(std::exp(-1.0) + 1 + std::exp(1.0))}) < 1000 * tolerance);
	REQUIRE(MaxDiff(TStat::LogSumExp(matrix, TMat::axisRows), {1000, 1000, 1000}) < 1000 * tolerance);

	const TVector<TestType> noRows = TStat::LogSumExp(TMat(0, 2), TMat::axisRows);
	REQUIRE(noRows.Size() == 2);
	REQUIRE(noRows[0] == -infinity);
}

TEMPLATE_TEST_CASE("Statistics softmax", "[statistics]", float, double)
{
	typedef TStatistics<TestType> TStat;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	const double e1 = std::exp(1.0);
	const double e2 = std::exp(2.0);
	const double sum = 1 + e1 + e2;

	REQUIRE(MaxDiff(TStat::Softmax(MakeVector<TestType>({0, 1, 2})), {1 / sum, e1 / sum, e2 / sum}) < tolerance);
	// softmax depends on the differences only, and does not overflow
	REQUIRE(MaxDiff(TStat::Softmax(MakeVector<TestType>({1000, 1001, 1002})), {1 / sum, e1 / sum, e2 / sum}) < tolerance);
	REQUIRE(MaxDiff(TStat::Softmax(MakeVector<TestType>({-1000, -999, -998})), {1 / sum, e1 / sum, e2 / sum}) < tolerance);

	REQUIRE(MaxDiff(TStat::Softmax(MakeVector<TestType>({7, 7, 7, 7})), {0.25, 0.25, 0.25, 0.25}) < tolerance);
	REQUIRE(MaxDiff(TStat::Softmax(MakeVector<TestType>({3})), {1}) == 0);
	REQUIRE(MaxDiff(TStat::Softmax(MakeVector<TestType>({0, 5000})), {0, 1}) == 0);
	REQUIRE(TStat::Softmax(TVector<TestType>()).Empty());
	REQUIRE(double(TStat::Softmax(MakeVector<TestType>({0.3, -1.2, 2.5, 0.1, 4})).Sum()) == Catch::Approx(1).margin(tolerance));

	// softmax(x)[i] = exp(x[i] - logsumexp(x))
	const TVector<TestType> x = MakeVector<TestType>({0.5, -2, 3.25, 1});
	const TVector<TestType> softmax = TStat::Softmax(x);
	const double logSumExp = double(TStat::LogSumExp(x));
	for (size_t i = 0; i < x.Size(); ++i)
		REQUIRE(double(softmax[i]) == Catch::Approx(std::exp(double(x[i]) - logSumExp)).margin(tolerance));

	// a matrix: every row (axisCols) or every column (axisRows) sums to 1
	const TMat matrix = MakeMatrix<TestType>(2, 3, {0, 1, 2, 1000, 1000, 1000});

	const TMat byRow = TStat::Softmax(matrix, TMat::axisCols);
	REQUIRE(byRow.Rows() == 2);
	REQUIRE(byRow.Cols() == 3);
	REQUIRE(MaxDiff(byRow, 2, 3, {1 / sum, e1 / sum, e2 / sum, 1 / 3.0, 1 / 3.0, 1 / 3.0}) < tolerance);
	REQUIRE(MaxDiff(byRow.Sum(TMat::axisCols), {1, 1}) < tolerance);

	const TMat byColumn = TStat::Softmax(matrix, TMat::axisRows);
	REQUIRE(MaxDiff(byColumn, 2, 3, {0, 0, 0, 1, 1, 1}) == 0);

	REQUIRE(TStat::Softmax(TMat(0, 3), TMat::axisCols).Rows() == 0);
}

TEST_CASE("Statistics keep the allocator of the input", "[statistics]")
{
	IAllocPtr alloc = MakeAlloc();

	{
		const CMatrix matrix = MakeMatrix<double>(3, 2, {1, 2, 3, 4, 5, 9}, alloc);
		const CVector vec = MakeVector<double>({1, 2, 3}, alloc);

		REQUIRE(CStatistics::Mean(matrix, CMatrix::axisRows).GetAlloc() == alloc);
		REQUIRE(CStatistics::Variance(matrix, CMatrix::axisRows).GetAlloc() == alloc);
		REQUIRE(CStatistics::Covariance(matrix).GetAlloc() == alloc);
		REQUIRE(CStatistics::LogSumExp(matrix, CMatrix::axisCols).GetAlloc() == alloc);
		REQUIRE(CStatistics::Softmax(matrix, CMatrix::axisCols).GetAlloc() == alloc);
		REQUIRE(CStatistics::Softmax(vec).GetAlloc() == alloc);
	}

	RequireNoLeaks(alloc);
}
