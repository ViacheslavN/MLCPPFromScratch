#include "MetricsUtils.h"
#include "ClassificationMetrics.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <string>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace evaluation
	{
		namespace detail
		{
			void CheckPair(size_t nTrue, size_t nOther, const char* pszMetric)
			{
				if (nTrue != nOther)
					throw CommonLib::CExcBase("{0}: the sizes {1} and {2} do not match", std::string(pszMetric), nTrue, nOther);

				if (nTrue == 0)
					throw CommonLib::CExcBase("{0}: no samples", std::string(pszMetric));
			}

			size_t ClassNumber(double label, size_t nClasses, const char* pszMetric)
			{
				// Written with !(...) so that NaN is rejected too.
				if (!(label >= 0 && label < 1e9 && label == std::floor(label)))
					throw CommonLib::CExcBase("{0}: label {1} is not a class number (0, 1, 2, ...)", std::string(pszMetric), label);

				const size_t nClass = size_t(label);
				if (nClasses != 0 && nClass >= nClasses)
					throw CommonLib::CExcBase("{0}: label {1}, but there are {2} classes", std::string(pszMetric), nClass, nClasses);

				return nClass;
			}

			void Throw(const char* pszMetric, const char* pszMessage)
			{
				throw CommonLib::CExcBase("{0}: {1}", std::string(pszMetric), std::string(pszMessage));
			}
		}

		using numerics::CMatrixI;
		using numerics::CVector;

		namespace
		{
			double Ratio(double numerator, double denominator)
			{
				return denominator > 0 ? numerator / denominator : 0.0;
			}

			double F1Of(double precision, double recall)
			{
				return Ratio(2 * precision * recall, precision + recall);
			}
		}

		// ---- Confusion matrix ----

		CMatrixI CClassificationMetrics::ConfusionMatrix(const std::vector<size_t>& classesTrue, const std::vector<size_t>& classesPredicted, size_t nClasses)
		{
			if (nClasses == 0)
			{
				for (size_t i = 0; i < classesTrue.size(); ++i)
					nClasses = (std::max)(nClasses, (std::max)(classesTrue[i], classesPredicted[i]) + 1);
			}

			CMatrixI confusion(nClasses, nClasses);
			for (size_t i = 0; i < classesTrue.size(); ++i)
				confusion(classesTrue[i], classesPredicted[i]) += 1;

			return confusion;
		}

		// ---- Precision, recall, F1 ----

		std::vector<SClassScores> CClassificationMetrics::ClassScores(const CMatrixI& confusion)
		{
			if (confusion.Rows() != confusion.Cols())
				throw CommonLib::CExcBase("Class scores: the confusion matrix must be square, size {0} x {1}", confusion.Rows(), confusion.Cols());

			const size_t nClasses = confusion.Rows();

			std::vector<SClassScores> scores(nClasses);
			for (size_t k = 0; k < nClasses; ++k)
			{
				// Row k: the true samples of the class. Column k: the samples predicted as it.
				double nTrue = 0;
				double nPredicted = 0;
				for (size_t j = 0; j < nClasses; ++j)
				{
					nTrue += double(confusion(k, j));
					nPredicted += double(confusion(j, k));
				}

				const double nCorrect = double(confusion(k, k));

				scores[k].precision = Ratio(nCorrect, nPredicted);
				scores[k].recall = Ratio(nCorrect, nTrue);
				scores[k].f1 = F1Of(scores[k].precision, scores[k].recall);
				scores[k].support = size_t(nTrue);
			}

			return scores;
		}

		SClassScores CClassificationMetrics::OneClass(const CMatrixI& confusion, size_t nClass, const char* pszMetric)
		{
			const std::vector<SClassScores> scores = ClassScores(confusion);

			// A class above the largest label: it has no samples and was never predicted.
			if (nClass >= scores.size())
				throw CommonLib::CExcBase("{0}: class {1} does not appear in the labels", std::string(pszMetric), nClass);

			return scores[nClass];
		}

		SClassScores CClassificationMetrics::Average(const CMatrixI& confusion, EAverage average)
		{
			const std::vector<SClassScores> scores = ClassScores(confusion);
			const size_t nClasses = scores.size();

			SClassScores result;

			if (average == averageMicro)
			{
				// Every sample is one "predicted" and one "true", so the totals of both
				// are the number of samples: precision = recall = accuracy.
				double nCorrect = 0;
				double nTotal = 0;
				for (size_t k = 0; k < nClasses; ++k)
				{
					nCorrect += double(confusion(k, k));
					nTotal += double(scores[k].support);
				}

				result.precision = Ratio(nCorrect, nTotal);
				result.recall = result.precision;
				result.f1 = F1Of(result.precision, result.recall);
				result.support = size_t(nTotal);
				return result;
			}

			double weightSum = 0;
			for (size_t k = 0; k < nClasses; ++k)
			{
				double nPredicted = 0;
				for (size_t j = 0; j < nClasses; ++j)
					nPredicted += double(confusion(j, k));

				// Macro: a class number that is not used at all does not count.
				const double weight = average == averageWeighted ? double(scores[k].support)
					: (scores[k].support > 0 || nPredicted > 0 ? 1.0 : 0.0);

				result.precision += weight * scores[k].precision;
				result.recall += weight * scores[k].recall;
				result.f1 += weight * scores[k].f1;
				result.support += scores[k].support;
				weightSum += weight;
			}

			result.precision = Ratio(result.precision, weightSum);
			result.recall = Ratio(result.recall, weightSum);
			result.f1 = Ratio(result.f1, weightSum);
			return result;
		}

		// ---- ROC curve and AUC ----

		SRocCurve CClassificationMetrics::RocCurve(const std::vector<size_t>& classes, const std::vector<double>& scores)
		{
			const size_t nCount = classes.size();

			size_t nPositive = 0;
			for (size_t i = 0; i < nCount; ++i)
			{
				if (std::isnan(scores[i]))
					throw CommonLib::CExcBase("ROC curve: the score of sample {0} is not a number", i);

				nPositive += classes[i];
			}

			const size_t nNegative = nCount - nPositive;
			if (nPositive == 0 || nNegative == 0)
				throw CommonLib::CExcBase("ROC curve: both classes are needed, there are {0} positive and {1} negative samples", nPositive, nNegative);

			// The samples from the highest score down. Lowering the threshold past a
			// score turns its samples positive: a true positive moves the curve up, a
			// false positive moves it right.
			std::vector<size_t> order(nCount);
			std::iota(order.begin(), order.end(), size_t(0));
			std::stable_sort(order.begin(), order.end(), [&scores](size_t a, size_t b) { return scores[a] > scores[b]; });

			std::vector<double> rateFalse(1, 0.0);
			std::vector<double> rateTrue(1, 0.0);
			std::vector<double> thresholds(1, std::numeric_limits<double>::infinity());

			size_t nTrue = 0;
			size_t nFalse = 0;
			for (size_t i = 0; i < nCount; ++i)
			{
				if (classes[order[i]] == 1)
					++nTrue;
				else
					++nFalse;

				// Samples with the same score cross the threshold together: one point.
				if (i + 1 < nCount && scores[order[i + 1]] == scores[order[i]])
					continue;

				rateFalse.push_back(double(nFalse) / double(nNegative));
				rateTrue.push_back(double(nTrue) / double(nPositive));
				thresholds.push_back(scores[order[i]]);
			}

			SRocCurve curve;
			curve.falsePositiveRate = CVector(rateFalse.size());
			curve.truePositiveRate = CVector(rateFalse.size());
			curve.thresholds = CVector(rateFalse.size());
			for (size_t i = 0; i < rateFalse.size(); ++i)
			{
				curve.falsePositiveRate[i] = rateFalse[i];
				curve.truePositiveRate[i] = rateTrue[i];
				curve.thresholds[i] = thresholds[i];
			}

			return curve;
		}

		double CClassificationMetrics::AUC(const SRocCurve& curve)
		{
			const size_t nCount = curve.falsePositiveRate.Size();
			if (nCount != curve.truePositiveRate.Size())
				throw CommonLib::CExcBase("AUC: the curve has {0} false positive rates and {1} true positive rates", nCount, curve.truePositiveRate.Size());

			// The area under the straight lines between the points (trapezoids).
			double area = 0;
			for (size_t i = 1; i < nCount; ++i)
			{
				area += (curve.falsePositiveRate[i] - curve.falsePositiveRate[i - 1])
					* (curve.truePositiveRate[i] + curve.truePositiveRate[i - 1]) / 2;
			}

			return area;
		}

		// ---- Log-loss ----

		double CClassificationMetrics::LogLossTerm(double probability)
		{
			const double epsilon = 1e-15;

			if (std::isnan(probability))
				throw CommonLib::CExcBase("Log-loss: a probability is not a number");

			if (probability < epsilon)
				probability = epsilon;
			else if (probability > 1 - epsilon)
				probability = 1 - epsilon;

			return -std::log(probability);
		}
	}
}
