#include "TestCommon.h"

using namespace ML::numerics;
using namespace ML::evaluation;
using namespace test_utils;

// A two-class example used by several tests: 4 true positives, 1 false negative,
// 2 false positives, 3 true negatives.
//   true:      1 1 1 1 1 0 0 0 0 0
//   predicted: 1 1 1 1 0 1 1 0 0 0
template<class T>
static TVector<T> BinaryTrue() { return MakeVector<T>({1, 1, 1, 1, 1, 0, 0, 0, 0, 0}); }
template<class T>
static TVector<T> BinaryPredicted() { return MakeVector<T>({1, 1, 1, 1, 0, 1, 1, 0, 0, 0}); }

// A three-class example:
//   true:      0 0 0 0 1 1 1 2 2 2
//   predicted: 0 0 1 2 1 1 0 2 2 2
// confusion: [2 1 1] [1 2 0] [0 0 3]
template<class T>
static TVector<T> ThreeTrue() { return MakeVector<T>({0, 0, 0, 0, 1, 1, 1, 2, 2, 2}); }
template<class T>
static TVector<T> ThreePredicted() { return MakeVector<T>({0, 0, 1, 2, 1, 1, 0, 2, 2, 2}); }

// ---------------------------------------------------------------------------
// Accuracy and confusion matrix
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Accuracy", "[metrics][classification]", int, float, double)
{
	REQUIRE(CClassificationMetrics::Accuracy(BinaryTrue<TestType>(), BinaryPredicted<TestType>()) == Near(0.7));
	REQUIRE(CClassificationMetrics::Accuracy(ThreeTrue<TestType>(), ThreePredicted<TestType>()) == Near(0.7));
	REQUIRE(CClassificationMetrics::Accuracy(ThreeTrue<TestType>(), ThreeTrue<TestType>()) == 1);
	REQUIRE(CClassificationMetrics::Accuracy(MakeVector<TestType>({0, 1}), MakeVector<TestType>({1, 0})) == 0);

	REQUIRE_THROWS_AS(CClassificationMetrics::Accuracy(TVector<TestType>(3), TVector<TestType>(2)), CExcBase);
	REQUIRE_THROWS_AS(CClassificationMetrics::Accuracy(TVector<TestType>(), TVector<TestType>()), CExcBase);
}

TEMPLATE_TEST_CASE("Confusion matrix", "[metrics][classification]", int, float, double)
{
	// rows: the true class; columns: the predicted class
	const CMatrixI binary = CClassificationMetrics::ConfusionMatrix(BinaryTrue<TestType>(), BinaryPredicted<TestType>());
	REQUIRE(binary.Rows() == 2);
	REQUIRE(binary.Cols() == 2);
	REQUIRE(Equals(binary, {3, 2, 1, 4}));		// TN FP / FN TP

	const CMatrixI three = CClassificationMetrics::ConfusionMatrix(ThreeTrue<TestType>(), ThreePredicted<TestType>());
	REQUIRE(Equals(three, {2, 1, 1, 1, 2, 0, 0, 0, 3}));
	REQUIRE(three.Sum() == 10);		// every sample is counted once

	// the number of classes can be given: a class that never occurs gets a zero row and column
	const CMatrixI four = CClassificationMetrics::ConfusionMatrix(ThreeTrue<TestType>(), ThreePredicted<TestType>(), 4);
	REQUIRE(four.Rows() == 4);
	REQUIRE(four(3, 3) == 0);
	REQUIRE(four.Sum() == 10);

	// a class only in the predictions still gets its column
	REQUIRE(Equals(CClassificationMetrics::ConfusionMatrix(MakeVector<TestType>({0, 0}), MakeVector<TestType>({0, 2})), {1, 0, 1, 0, 0, 0, 0, 0, 0}));

	REQUIRE_THROWS_AS(CClassificationMetrics::ConfusionMatrix(MakeVector<TestType>({0, -1}), MakeVector<TestType>({0, 1})), CExcBase);		// negative label
	REQUIRE_THROWS_AS(CClassificationMetrics::ConfusionMatrix(ThreeTrue<TestType>(), ThreePredicted<TestType>(), 2), CExcBase);			// label 2 of 2 classes
	REQUIRE_THROWS_AS(CClassificationMetrics::ConfusionMatrix(TVector<TestType>(3), TVector<TestType>(2)), CExcBase);
}

TEST_CASE("Confusion matrix rejects labels that are not whole numbers", "[metrics][classification]")
{
	REQUIRE_THROWS_AS(CClassificationMetrics::ConfusionMatrix(MakeVector<double>({0, 0.5}), MakeVector<double>({0, 1})), CExcBase);

	const double nan = std::numeric_limits<double>::quiet_NaN();
	REQUIRE_THROWS_AS(CClassificationMetrics::ConfusionMatrix(MakeVector<double>({0, nan}), MakeVector<double>({0, 1})), CExcBase);
}

// ---------------------------------------------------------------------------
// Precision, recall, F1
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Precision, recall and F1 of two classes", "[metrics][classification]", int, float, double)
{
	const TVector<TestType> yTrue = BinaryTrue<TestType>();
	const TVector<TestType> yPredicted = BinaryPredicted<TestType>();

	// the positive class 1: TP 4, FP 2, FN 1
	REQUIRE(CClassificationMetrics::Precision(yTrue, yPredicted) == Near(4.0 / 6));
	REQUIRE(CClassificationMetrics::Recall(yTrue, yPredicted) == Near(4.0 / 5));
	REQUIRE(CClassificationMetrics::F1(yTrue, yPredicted) == Near(2 * 4.0 / (2 * 4 + 2 + 1)));		// 2TP / (2TP + FP + FN)

	// class 0 as the positive one: TP 3, FP 1, FN 2
	REQUIRE(CClassificationMetrics::Precision(yTrue, yPredicted, 0) == Near(3.0 / 4));
	REQUIRE(CClassificationMetrics::Recall(yTrue, yPredicted, 0) == Near(3.0 / 5));

	const std::vector<SClassScores> scores = CClassificationMetrics::ClassScores(yTrue, yPredicted);
	REQUIRE(scores.size() == 2);
	REQUIRE(scores[0].support == 5);
	REQUIRE(scores[1].support == 5);
	REQUIRE(scores[1].precision == Near(4.0 / 6));
	REQUIRE(scores[1].recall == Near(0.8));

	REQUIRE_THROWS_AS(CClassificationMetrics::Precision(yTrue, yPredicted, 2), CExcBase);		// no such class
}

TEMPLATE_TEST_CASE("Precision and recall pull in opposite directions", "[metrics][classification]", int, double)
{
	const TVector<TestType> yTrue = MakeVector<TestType>({1, 1, 1, 1, 0, 0, 0, 0, 0, 0});

	// everything predicted positive: every positive is found, most predictions are wrong
	const TVector<TestType> allPositive(10, TestType(1));
	REQUIRE(CClassificationMetrics::Recall(yTrue, allPositive) == 1);
	REQUIRE(CClassificationMetrics::Precision(yTrue, allPositive) == Near(0.4));

	// one sure positive: no wrong prediction, most positives are missed
	const TVector<TestType> onePositive = MakeVector<TestType>({1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
	REQUIRE(CClassificationMetrics::Precision(yTrue, onePositive) == 1);
	REQUIRE(CClassificationMetrics::Recall(yTrue, onePositive) == Near(0.25));
	REQUIRE(CClassificationMetrics::F1(yTrue, onePositive) == Near(0.4));

	// nothing predicted positive: the ratios with an empty denominator are 0
	const TVector<TestType> nonePositive = MakeVector<TestType>({0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
	REQUIRE(CClassificationMetrics::Precision(yTrue, nonePositive) == 0);
	REQUIRE(CClassificationMetrics::Recall(yTrue, nonePositive) == 0);
	REQUIRE(CClassificationMetrics::F1(yTrue, nonePositive) == 0);
}

TEMPLATE_TEST_CASE("Precision, recall and F1 of three classes", "[metrics][classification]", int, float, double)
{
	const TVector<TestType> yTrue = ThreeTrue<TestType>();
	const TVector<TestType> yPredicted = ThreePredicted<TestType>();

	// confusion [2 1 1] [1 2 0] [0 0 3]
	// class 0: precision 2/3, recall 2/4; class 1: 2/3, 2/3; class 2: 3/4, 3/3
	const std::vector<SClassScores> scores = CClassificationMetrics::ClassScores(yTrue, yPredicted);
	REQUIRE(scores.size() == 3);
	REQUIRE(scores[0].precision == Near(2.0 / 3));
	REQUIRE(scores[0].recall == Near(0.5));
	REQUIRE(scores[0].support == 4);
	REQUIRE(scores[1].precision == Near(2.0 / 3));
	REQUIRE(scores[1].recall == Near(2.0 / 3));
	REQUIRE(scores[1].f1 == Near(2.0 / 3));
	REQUIRE(scores[2].precision == Near(0.75));
	REQUIRE(scores[2].recall == 1);
	REQUIRE(scores[2].f1 == Near(2 * 0.75 / 1.75));

	REQUIRE(CClassificationMetrics::Precision(yTrue, yPredicted, 2) == Near(0.75));
	REQUIRE(CClassificationMetrics::Recall(yTrue, yPredicted, 0) == Near(0.5));

	const double f0 = 2 * (2.0 / 3) * 0.5 / (2.0 / 3 + 0.5);
	const double f1 = 2.0 / 3;
	const double f2 = 2 * 0.75 / 1.75;

	REQUIRE(CClassificationMetrics::Precision(yTrue, yPredicted, averageMacro) == Near((2.0 / 3 + 2.0 / 3 + 0.75) / 3));
	REQUIRE(CClassificationMetrics::Recall(yTrue, yPredicted, averageMacro) == Near((0.5 + 2.0 / 3 + 1) / 3));
	REQUIRE(CClassificationMetrics::F1(yTrue, yPredicted, averageMacro) == Near((f0 + f1 + f2) / 3));

	// weighted by the supports 4, 3, 3; the weighted recall is the accuracy
	REQUIRE(CClassificationMetrics::Precision(yTrue, yPredicted, averageWeighted) == Near((4 * 2.0 / 3 + 3 * 2.0 / 3 + 3 * 0.75) / 10));
	REQUIRE(CClassificationMetrics::Recall(yTrue, yPredicted, averageWeighted) == Near(0.7));
	REQUIRE(CClassificationMetrics::F1(yTrue, yPredicted, averageWeighted) == Near((4 * f0 + 3 * f1 + 3 * f2) / 10));

	// micro: all three equal the accuracy
	REQUIRE(CClassificationMetrics::Precision(yTrue, yPredicted, averageMicro) == Near(0.7));
	REQUIRE(CClassificationMetrics::Recall(yTrue, yPredicted, averageMicro) == Near(0.7));
	REQUIRE(CClassificationMetrics::F1(yTrue, yPredicted, averageMicro) == Near(0.7));
}

TEST_CASE("Macro average leaves out class numbers that are not used", "[metrics][classification]")
{
	// the labels are 0 and 2: class 1 never occurs
	const CVectorI yTrue = MakeVector<int>({0, 0, 2, 2});
	const CVectorI yPredicted = MakeVector<int>({0, 2, 2, 2});

	// class 0: precision 1, recall 0.5; class 2: precision 2/3, recall 1
	REQUIRE(CClassificationMetrics::Precision(yTrue, yPredicted, averageMacro) == Near((1 + 2.0 / 3) / 2));
	REQUIRE(CClassificationMetrics::Recall(yTrue, yPredicted, averageMacro) == Near(0.75));
}

TEST_CASE("Class scores from a confusion matrix", "[metrics][classification]")
{
	const std::vector<SClassScores> scores = CClassificationMetrics::ClassScores(MakeMatrix<int>(2, 2, {3, 2, 1, 4}));
	REQUIRE(scores[1].precision == Near(4.0 / 6));
	REQUIRE(scores[1].recall == Near(0.8));
	REQUIRE(scores[0].support == 5);

	REQUIRE_THROWS_AS(CClassificationMetrics::ClassScores(MakeMatrix<int>(2, 3, {1, 2, 3, 4, 5, 6})), CExcBase);
}

// ---------------------------------------------------------------------------
// ROC curve and AUC
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("ROC curve of a small example", "[metrics][classification][roc]", float, double)
{
	// the textbook example: 2 negatives, 2 positives
	const CVectorI yTrue = MakeVector<int>({0, 0, 1, 1});
	const TVector<TestType> scores = MakeVector<TestType>({0.1, 0.4, 0.35, 0.8});

	const SRocCurve curve = CClassificationMetrics::RocCurve(yTrue, scores);

	// thresholds +inf, 0.8, 0.4, 0.35, 0.1
	REQUIRE(curve.thresholds.Size() == 5);
	REQUIRE(curve.thresholds[0] == std::numeric_limits<double>::infinity());
	REQUIRE(curve.thresholds[1] == double(TestType(0.8)));
	REQUIRE(curve.thresholds[4] == double(TestType(0.1)));
	REQUIRE(MaxDiff(curve.falsePositiveRate, {0, 0, 0.5, 0.5, 1}) == 0);
	REQUIRE(MaxDiff(curve.truePositiveRate, {0, 0.5, 0.5, 1, 1}) == 0);

	REQUIRE(CClassificationMetrics::AUC(curve) == Near(0.75));
	REQUIRE(CClassificationMetrics::AUC(yTrue, scores) == Near(0.75));
}

TEMPLATE_TEST_CASE("AUC", "[metrics][classification][roc]", float, double)
{
	const CVectorI yTrue = MakeVector<int>({0, 0, 0, 1, 1, 1});

	// every positive above every negative: perfect
	REQUIRE(CClassificationMetrics::AUC(yTrue, MakeVector<TestType>({0.1, 0.2, 0.3, 0.7, 0.8, 0.9})) == 1);
	// every positive below every negative: perfectly wrong
	REQUIRE(CClassificationMetrics::AUC(yTrue, MakeVector<TestType>({0.7, 0.8, 0.9, 0.1, 0.2, 0.3})) == 0);
	// the same score for everyone: no information, one diagonal step
	const SRocCurve flat = CClassificationMetrics::RocCurve(yTrue, TVector<TestType>(6, TestType(0.5)));
	REQUIRE(flat.thresholds.Size() == 2);
	REQUIRE(CClassificationMetrics::AUC(flat) == Near(0.5));

	// only the order of the scores matters, not their scale: decision values work too
	const TVector<TestType> probabilities = MakeVector<TestType>({0.2, 0.6, 0.3, 0.5, 0.9, 0.4});
	const TVector<TestType> decisions = MakeVector<TestType>({-30, 10, -20, 0, 500, -10});
	REQUIRE(CClassificationMetrics::AUC(yTrue, probabilities) == CClassificationMetrics::AUC(yTrue, decisions));

	// AUC is the share of (positive, negative) pairs in the right order, ties count half:
	// positives 0.5, 0.9, 0.4 against negatives 0.2, 0.6, 0.3 -> 7 of 9 pairs
	REQUIRE(CClassificationMetrics::AUC(yTrue, probabilities) == Near(7.0 / 9));
	REQUIRE(CClassificationMetrics::AUC(MakeVector<int>({0, 1, 1}), MakeVector<TestType>({0.5, 0.5, 0.9})) == Near(0.75));
}

TEST_CASE("AUC equals the share of correctly ordered pairs on random data", "[metrics][classification][roc]")
{
	CRandom random;
	const size_t nCount = 300;

	CVectorI yTrue(nCount);
	CVector scores(nCount);
	for (size_t i = 0; i < nCount; ++i)
	{
		yTrue[i] = random.Next() > 0 ? 1 : 0;
		// rounded to one decimal: many ties; the positives score higher on average
		scores[i] = std::floor((random.Next() + 0.4 * yTrue[i]) * 10) / 10;
	}

	double ordered = 0;
	double pairs = 0;
	for (size_t i = 0; i < nCount; ++i)
	{
		for (size_t k = 0; k < nCount; ++k)
		{
			if (yTrue[i] != 1 || yTrue[k] != 0)
				continue;

			pairs += 1;
			if (scores[i] > scores[k])
				ordered += 1;
			else if (scores[i] == scores[k])
				ordered += 0.5;
		}
	}

	const double auc = CClassificationMetrics::AUC(yTrue, scores);
	REQUIRE(auc == Near(ordered / pairs));
	REQUIRE(auc > 0.55);
	REQUIRE(auc < 0.9);

	// the curve goes from (0, 0) to (1, 1) and never goes back
	const SRocCurve curve = CClassificationMetrics::RocCurve(yTrue, scores);
	const size_t nPoints = curve.thresholds.Size();
	REQUIRE(curve.falsePositiveRate[0] == 0);
	REQUIRE(curve.truePositiveRate[0] == 0);
	REQUIRE(curve.falsePositiveRate[nPoints - 1] == 1);
	REQUIRE(curve.truePositiveRate[nPoints - 1] == 1);
	for (size_t i = 1; i < nPoints; ++i)
	{
		REQUIRE(curve.falsePositiveRate[i] >= curve.falsePositiveRate[i - 1]);
		REQUIRE(curve.truePositiveRate[i] >= curve.truePositiveRate[i - 1]);
		REQUIRE(curve.thresholds[i] < curve.thresholds[i - 1]);
	}
}

TEST_CASE("ROC curve rejects bad input", "[metrics][classification][roc]")
{
	const CVector scores = MakeVector<double>({0.1, 0.4, 0.35});

	REQUIRE_THROWS_AS(CClassificationMetrics::RocCurve(MakeVector<int>({1, 1, 1}), scores), CExcBase);		// no negatives
	REQUIRE_THROWS_AS(CClassificationMetrics::RocCurve(MakeVector<int>({0, 0, 0}), scores), CExcBase);		// no positives
	REQUIRE_THROWS_AS(CClassificationMetrics::RocCurve(MakeVector<int>({0, 1, 2}), scores), CExcBase);		// three classes
	REQUIRE_THROWS_AS(CClassificationMetrics::RocCurve(MakeVector<int>({0, 1}), scores), CExcBase);			// sizes
	REQUIRE_THROWS_AS(CClassificationMetrics::AUC(MakeVector<int>({0, 1, 2}), scores), CExcBase);

	const double nan = std::numeric_limits<double>::quiet_NaN();
	REQUIRE_THROWS_AS(CClassificationMetrics::RocCurve(MakeVector<int>({0, 1, 1}), MakeVector<double>({0.1, nan, 0.3})), CExcBase);
}

// ---------------------------------------------------------------------------
// Log-loss
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Log-loss of two classes", "[metrics][classification][logloss]", float, double)
{
	const CVectorI yTrue = MakeVector<int>({1, 0, 1, 0});
	const TVector<TestType> probabilities = MakeVector<TestType>({0.9, 0.1, 0.8, 0.35});

	// -log of the probability given to the true class: 0.9, 0.9, 0.8, 0.65
	const double expected = -(std::log(double(TestType(0.9))) + std::log(1 - double(TestType(0.1)))
		+ std::log(double(TestType(0.8))) + std::log(1 - double(TestType(0.35)))) / 4;
	REQUIRE(CClassificationMetrics::LogLoss(yTrue, probabilities) == Near(expected));

	// knowing nothing (0.5 everywhere) costs log(2)
	REQUIRE(CClassificationMetrics::LogLoss(yTrue, TVector<TestType>(4, TestType(0.5))) == Near(std::log(2.0)));

	// confident and right is close to 0; confident and wrong is large but finite
	REQUIRE(CClassificationMetrics::LogLoss(yTrue, MakeVector<TestType>({1, 0, 1, 0})) == Catch::Approx(0).margin(1e-14));
	const double wrong = CClassificationMetrics::LogLoss(yTrue, MakeVector<TestType>({0, 1, 0, 1}));
	REQUIRE(wrong == Near(-std::log(1e-15)));
	REQUIRE(std::isfinite(wrong));

	REQUIRE_THROWS_AS(CClassificationMetrics::LogLoss(MakeVector<int>({0, 1, 2, 0}), probabilities), CExcBase);		// three classes
	REQUIRE_THROWS_AS(CClassificationMetrics::LogLoss(MakeVector<int>({0, 1}), probabilities), CExcBase);
}

TEMPLATE_TEST_CASE("Log-loss of several classes", "[metrics][classification][logloss]", float, double)
{
	const CVectorI yTrue = MakeVector<int>({0, 2, 1});
	const TMatrix<TestType> probabilities = MakeMatrix<TestType>(3, 3, {0.7, 0.2, 0.1, 0.1, 0.3, 0.6, 0.25, 0.5, 0.25});

	const double expected = -(std::log(double(TestType(0.7))) + std::log(double(TestType(0.6))) + std::log(0.5)) / 3;
	REQUIRE(CClassificationMetrics::LogLoss(yTrue, probabilities) == Near(expected));

	// with two columns it is the two-class log-loss
	const CVectorI binary = MakeVector<int>({1, 0, 1});
	const TMatrix<TestType> twoColumns = MakeMatrix<TestType>(3, 2, {0.25, 0.75, 0.5, 0.5, 0.875, 0.125});
	REQUIRE(CClassificationMetrics::LogLoss(binary, twoColumns)
		== Near(CClassificationMetrics::LogLoss(binary, MakeVector<TestType>({0.75, 0.5, 0.125}))));

	// uniform probabilities over k classes cost log(k)
	REQUIRE(CClassificationMetrics::LogLoss(yTrue, TMatrix<TestType>(3, 3, TestType(1.0 / 3))) == Catch::Approx(std::log(3.0)).margin(1e-6));

	REQUIRE_THROWS_AS(CClassificationMetrics::LogLoss(MakeVector<int>({0, 3, 1}), probabilities), CExcBase);		// label 3 of 3 classes
	REQUIRE_THROWS_AS(CClassificationMetrics::LogLoss(MakeVector<int>({0, 1}), probabilities), CExcBase);
}
