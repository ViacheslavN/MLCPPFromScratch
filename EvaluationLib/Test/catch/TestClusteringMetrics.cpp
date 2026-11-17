#include "TestCommon.h"

using namespace ML::numerics;
using namespace ML::evaluation;
using namespace test_utils;

// ---------------------------------------------------------------------------
// Inertia
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Inertia", "[metrics][clustering]", float, double)
{
	// two clusters on a line: {0, 2} (center 1) and {10, 14} (center 12)
	const TMatrix<TestType> x = MakeMatrix<TestType>(4, 1, {0, 2, 10, 14});
	const CVectorI labels = MakeVector<int>({0, 0, 1, 1});

	// 1 + 1 + 4 + 4
	REQUIRE(CClusteringMetrics::Inertia(x, labels) == Near(10));

	// with given centers
	REQUIRE(CClusteringMetrics::Inertia(x, labels, MakeMatrix<TestType>(2, 1, {1, 12})) == Near(10));
	// centers that are not the means are worse
	REQUIRE(CClusteringMetrics::Inertia(x, labels, MakeMatrix<TestType>(2, 1, {0, 10})) == Near(4 + 16));

	// one cluster for everything: the sum of squares around the overall mean 6.5
	REQUIRE(CClusteringMetrics::Inertia(x, CVectorI(4)) == Near(42.25 + 20.25 + 12.25 + 56.25));
	// every sample its own cluster: 0
	REQUIRE(CClusteringMetrics::Inertia(x, MakeVector<int>({0, 1, 2, 3})) == 0);

	// two features: the squared distances add up over the features
	const TMatrix<TestType> plane = MakeMatrix<TestType>(4, 2, {0, 0, 2, 2, 10, 0, 10, 4});
	REQUIRE(CClusteringMetrics::Inertia(plane, labels) == Near(2 + 2 + 4 + 4));

	// the labels can be any values when the centers are computed
	REQUIRE(CClusteringMetrics::Inertia(x, MakeVector<int>({7, 7, -3, -3})) == Near(10));
	REQUIRE(CClusteringMetrics::Inertia(x, MakeVector<double>({0.5, 0.5, 2.5, 2.5})) == Near(10));

	REQUIRE_THROWS_AS(CClusteringMetrics::Inertia(x, MakeVector<int>({0, 1})), CExcBase);											// sizes
	REQUIRE_THROWS_AS(CClusteringMetrics::Inertia(x, MakeVector<int>({0, 0, 1, 2}), MakeMatrix<TestType>(2, 1, {1, 12})), CExcBase);	// no center 2
	REQUIRE_THROWS_AS(CClusteringMetrics::Inertia(x, labels, MakeMatrix<TestType>(2, 2, {1, 0, 12, 0})), CExcBase);					// features
	REQUIRE_THROWS_AS(CClusteringMetrics::Inertia(TMatrix<TestType>(0, 1), CVectorI()), CExcBase);
}

// ---------------------------------------------------------------------------
// Silhouette
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Silhouette of a small example", "[metrics][clustering][silhouette]", float, double)
{
	// {0, 1} and {10, 11} on a line
	const TMatrix<TestType> x = MakeMatrix<TestType>(4, 1, {0, 1, 10, 11});
	const CVectorI labels = MakeVector<int>({0, 0, 1, 1});

	// sample 0: a = 1, b = (10 + 11) / 2 = 10.5 -> 9.5 / 10.5
	// sample 1: a = 1, b = (9 + 10) / 2 = 9.5  -> 8.5 / 9.5
	const CVector samples = CClusteringMetrics::SilhouetteSamples(x, labels);
	REQUIRE(samples.Size() == 4);
	REQUIRE(samples[0] == Near(9.5 / 10.5));
	REQUIRE(samples[1] == Near(8.5 / 9.5));
	REQUIRE(samples[2] == Near(8.5 / 9.5));
	REQUIRE(samples[3] == Near(9.5 / 10.5));

	REQUIRE(CClusteringMetrics::SilhouetteScore(x, labels) == Near((9.5 / 10.5 + 8.5 / 9.5) / 2));

	// the same clustering with other label values
	REQUIRE(CClusteringMetrics::SilhouetteScore(x, MakeVector<int>({5, 5, 2, 2})) == Near((9.5 / 10.5 + 8.5 / 9.5) / 2));

	// a wrong clustering of the same data: {0, 10} and {1, 11}
	const double wrong = CClusteringMetrics::SilhouetteScore(x, MakeVector<int>({0, 1, 0, 1}));
	REQUIRE(wrong < 0);
}

TEMPLATE_TEST_CASE("Silhouette special cases", "[metrics][clustering][silhouette]", float, double)
{
	const TMatrix<TestType> x = MakeMatrix<TestType>(5, 1, {0, 1, 2, 10, 20});

	// a sample alone in its cluster gets 0
	const CVector samples = CClusteringMetrics::SilhouetteSamples(x, MakeVector<int>({0, 0, 0, 1, 2}));
	REQUIRE(samples[3] == 0);
	REQUIRE(samples[4] == 0);
	REQUIRE(samples[0] > 0.8);

	// identical samples in two clusters: a = b = 0
	const TMatrix<TestType> same(4, 2, TestType(3));
	REQUIRE(CClusteringMetrics::SilhouetteScore(same, MakeVector<int>({0, 0, 1, 1})) == 0);

	// every value is in [-1, 1]
	CRandom random;
	const TMatrix<TestType> points = random.Matrix<TestType>(60, 3);
	CVectorI labels(60);
	for (size_t i = 0; i < 60; ++i)
		labels[i] = int(i % 4);

	const CVector values = CClusteringMetrics::SilhouetteSamples(points, labels);
	for (size_t i = 0; i < values.Size(); ++i)
	{
		REQUIRE(values[i] >= -1);
		REQUIRE(values[i] <= 1);
	}

	// random labels on random points: no structure, a score near 0
	REQUIRE(std::fabs(CClusteringMetrics::SilhouetteScore(points, labels)) < 0.15);

	REQUIRE_THROWS_AS(CClusteringMetrics::SilhouetteScore(x, CVectorI(5)), CExcBase);							// one cluster
	REQUIRE_THROWS_AS(CClusteringMetrics::SilhouetteScore(x, MakeVector<int>({0, 1, 2, 3, 4})), CExcBase);	// as many clusters as samples
	REQUIRE_THROWS_AS(CClusteringMetrics::SilhouetteScore(x, MakeVector<int>({0, 1})), CExcBase);				// sizes
}

TEST_CASE("Silhouette is high for separated clusters and finds the right number", "[metrics][clustering][silhouette]")
{
	// three tight groups of 20 points around (0, 0), (10, 0) and (0, 10)
	CRandom random;
	const size_t nCount = 60;

	CMatrix x(nCount, 2);
	CVectorI three(nCount);		// the true grouping
	CVectorI two(nCount);		// groups 1 and 2 merged
	CVectorI six(nCount);		// every group cut in two at random
	for (size_t i = 0; i < nCount; ++i)
	{
		const int nGroup = int(i % 3);
		x(i, 0) = (nGroup == 1 ? 10.0 : 0.0) + random.Next();
		x(i, 1) = (nGroup == 2 ? 10.0 : 0.0) + random.Next();

		three[i] = nGroup;
		two[i] = nGroup == 0 ? 0 : 1;
		six[i] = nGroup * 2 + int((i / 3) % 2);
	}

	const double scoreThree = CClusteringMetrics::SilhouetteScore(x, three);
	REQUIRE(scoreThree > 0.85);
	REQUIRE(scoreThree > CClusteringMetrics::SilhouetteScore(x, two));
	REQUIRE(scoreThree > CClusteringMetrics::SilhouetteScore(x, six));

	// the inertia, in contrast, keeps falling as clusters are added
	const double inertiaTwo = CClusteringMetrics::Inertia(x, two);
	const double inertiaThree = CClusteringMetrics::Inertia(x, three);
	const double inertiaSix = CClusteringMetrics::Inertia(x, six);
	REQUIRE(inertiaTwo > inertiaThree);
	REQUIRE(inertiaThree > inertiaSix);
}
