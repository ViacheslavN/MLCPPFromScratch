#pragma once

#include <cmath>
#include <vector>

#include "MetricsUtils.h"

namespace ML
{
	namespace evaluation
	{
		// How the per-class precision, recall and F1 are combined into one number.
		enum EAverage
		{
			// The plain mean over the classes: every class counts the same, however rare.
			averageMacro,
			// The mean weighted by the number of true samples of each class.
			averageWeighted,
			// From the totals over all classes. With one label per sample this equals
			// the accuracy, for precision, recall and F1 alike.
			averageMicro
		};

		// Precision, recall and F1 of one class.
		struct SClassScores
		{
			// Of the samples predicted as this class, the share that really are it.
			double precision = 0;
			// Of the samples that really are this class, the share that were found.
			double recall = 0;
			// The harmonic mean of the two: 2 * precision * recall / (precision + recall).
			double f1 = 0;
			// The number of true samples of this class.
			size_t support = 0;
		};

		// The ROC curve: one point per threshold, from (0, 0) to (1, 1).
		struct SRocCurve
		{
			// Of the negative samples, the share with a score at or above the threshold.
			numerics::CVector falsePositiveRate;
			// Of the positive samples, the share with a score at or above the threshold.
			numerics::CVector truePositiveRate;
			// The thresholds, decreasing. The first one is +infinity: nothing is positive.
			numerics::CVector thresholds;
		};

		// How good predicted classes, scores and probabilities are. Static methods only.
		//
		// The methods are templates: the labels are a vector of any element type (int,
		// float, double), the scores and probabilities too, deduced from the arguments.
		// The classes are numbered 0, 1, ..., k - 1. The results are doubles.
		// Different sizes, empty data and labels that are not class numbers throw
		// CommonLib::CExcBase.
		class CClassificationMetrics
		{
		public:
			// ---- Accuracy ----

			// The share of the samples whose predicted class is the true one.
			template<class TLabel>
			static double Accuracy(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TLabel>& yPredicted)
			{
				detail::CheckPair(yTrue.Size(), yPredicted.Size(), "Accuracy");

				size_t nCorrect = 0;
				for (size_t i = 0; i < yTrue.Size(); ++i)
				{
					if (yTrue[i] == yPredicted[i])
						++nCorrect;
				}

				return double(nCorrect) / double(yTrue.Size());
			}

			// ---- Confusion matrix ----

			// classes x classes: element (t, p) is the number of samples of the true
			// class t that were predicted as class p. The diagonal holds the correct
			// predictions. nClasses == 0: the largest label plus one.
			template<class TLabel>
			static numerics::CMatrixI ConfusionMatrix(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TLabel>& yPredicted,
				size_t nClasses = 0)
			{
				detail::CheckPair(yTrue.Size(), yPredicted.Size(), "Confusion matrix");

				return ConfusionMatrix(Classes(yTrue, nClasses, "Confusion matrix"), Classes(yPredicted, nClasses, "Confusion matrix"), nClasses);
			}

			// ---- Precision, recall, F1 ----

			// The scores of every class, from a confusion matrix or from the labels.
			// A ratio with nothing in its denominator (a class never predicted, or with
			// no true samples) is 0.
			static std::vector<SClassScores> ClassScores(const numerics::CMatrixI& confusion);

			template<class TLabel>
			static std::vector<SClassScores> ClassScores(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TLabel>& yPredicted)
			{
				return ClassScores(ConfusionMatrix(yTrue, yPredicted));
			}

			// The score of one class against all the others. The default class 1 is the
			// positive class of a two-class problem.
			template<class TLabel>
			static double Precision(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TLabel>& yPredicted, size_t nClass = 1)
			{
				return OneClass(ConfusionMatrix(yTrue, yPredicted), nClass, "Precision").precision;
			}

			template<class TLabel>
			static double Recall(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TLabel>& yPredicted, size_t nClass = 1)
			{
				return OneClass(ConfusionMatrix(yTrue, yPredicted), nClass, "Recall").recall;
			}

			template<class TLabel>
			static double F1(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TLabel>& yPredicted, size_t nClass = 1)
			{
				return OneClass(ConfusionMatrix(yTrue, yPredicted), nClass, "F1").f1;
			}

			// One number for all the classes (see EAverage). Classes that appear neither
			// in 'yTrue' nor in 'yPredicted' are left out of the macro average.
			template<class TLabel>
			static double Precision(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TLabel>& yPredicted, EAverage average)
			{
				return Average(ConfusionMatrix(yTrue, yPredicted), average).precision;
			}

			template<class TLabel>
			static double Recall(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TLabel>& yPredicted, EAverage average)
			{
				return Average(ConfusionMatrix(yTrue, yPredicted), average).recall;
			}

			template<class TLabel>
			static double F1(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TLabel>& yPredicted, EAverage average)
			{
				return Average(ConfusionMatrix(yTrue, yPredicted), average).f1;
			}

			// ---- ROC curve and AUC (two classes) ----

			// 'yTrue' holds the labels 0 (negative) and 1 (positive), 'scores' a number
			// per sample that is larger the more likely the sample is positive: the
			// probability of class 1, or the decision value of an SVM. The curve shows
			// the trade-off for every possible threshold on the score. Throws when
			// 'yTrue' does not have both classes.
			template<class TLabel, class TScore>
			static SRocCurve RocCurve(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TScore>& scores)
			{
				detail::CheckPair(yTrue.Size(), scores.Size(), "ROC curve");

				std::vector<double> values(scores.Size());
				for (size_t i = 0; i < scores.Size(); ++i)
					values[i] = double(scores[i]);

				return RocCurve(Classes(yTrue, 2, "ROC curve"), values);
			}

			// The area under the ROC curve: the probability that a random positive
			// sample gets a higher score than a random negative one. 1 is a perfect
			// ranking, 0.5 is no better than chance.
			static double AUC(const SRocCurve& curve);

			template<class TLabel, class TScore>
			static double AUC(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TScore>& scores)
			{
				return AUC(RocCurve(yTrue, scores));
			}

			// ---- Log-loss (cross-entropy) ----

			// The mean of -log(probability given to the true class): 0 for confident
			// correct predictions, large for confident wrong ones. Probabilities are
			// clipped to [1e-15, 1 - 1e-15], so a wrong probability of exactly 0 gives
			// a large number, not infinity.

			// Two classes: 'yTrue' holds 0 and 1, 'probabilities' the probability of class 1.
			template<class TLabel, class TScore>
			static double LogLoss(const numerics::TVector<TLabel>& yTrue, const numerics::TVector<TScore>& probabilities)
			{
				detail::CheckPair(yTrue.Size(), probabilities.Size(), "Log-loss");

				const std::vector<size_t> classes = Classes(yTrue, 2, "Log-loss");

				double sum = 0;
				for (size_t i = 0; i < classes.size(); ++i)
					sum += LogLossTerm(classes[i] == 1 ? double(probabilities[i]) : 1 - double(probabilities[i]));

				return sum / double(classes.size());
			}

			// Any number of classes: 'probabilities' is samples x classes, element
			// (i, k) is the probability of class k for sample i.
			template<class TLabel, class TScore>
			static double LogLoss(const numerics::TVector<TLabel>& yTrue, const numerics::TMatrix<TScore>& probabilities)
			{
				detail::CheckPair(yTrue.Size(), probabilities.Rows(), "Log-loss");

				const std::vector<size_t> classes = Classes(yTrue, probabilities.Cols(), "Log-loss");

				double sum = 0;
				for (size_t i = 0; i < classes.size(); ++i)
					sum += LogLossTerm(double(probabilities(i, classes[i])));

				return sum / double(classes.size());
			}

		private:
			// The labels as class numbers, checked. nClasses != 0: every label is below it.
			template<class TLabel>
			static std::vector<size_t> Classes(const numerics::TVector<TLabel>& y, size_t nClasses, const char* pszMetric)
			{
				std::vector<size_t> classes(y.Size());
				for (size_t i = 0; i < y.Size(); ++i)
					classes[i] = detail::ClassNumber(double(y[i]), nClasses, pszMetric);

				return classes;
			}

			static numerics::CMatrixI ConfusionMatrix(const std::vector<size_t>& classesTrue, const std::vector<size_t>& classesPredicted, size_t nClasses);
			static SClassScores OneClass(const numerics::CMatrixI& confusion, size_t nClass, const char* pszMetric);
			static SClassScores Average(const numerics::CMatrixI& confusion, EAverage average);
			static SRocCurve RocCurve(const std::vector<size_t>& classes, const std::vector<double>& scores);
			static double LogLossTerm(double probability);
		};
	}
}
