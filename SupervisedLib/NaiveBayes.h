#pragma once

#include "Model.h"

namespace ML
{
	namespace supervised
	{
		// Gaussian Naive Bayes. For every class it learns how common the class is (the
		// prior) and, for every feature, the mean and the variance of the feature inside
		// the class. A row gets the class with the largest
		//   prior * product over the features of normal_density(value; mean, variance),
		// that is, Bayes' rule under the "naive" assumption that the features are
		// independent inside a class. Needs no scaling and no iterations.
		class CGaussianNaiveBayes : public IClassifier
		{
		public:
			// A share of the largest feature variance, added to every variance: keeps a
			// feature that is constant inside a class from producing a zero variance.
			explicit CGaussianNaiveBayes(double varianceSmoothing = 1e-9);

			void Fit(const CMatrix& x, const CVectorI& y) override;
			CVectorI Predict(const CMatrix& x) const override;
			// Rows() x classes: the probability of every class for every row; a row sums to 1.
			CMatrix PredictProbability(const CMatrix& x) const;

			// The share of the training samples in every class.
			const CVector& GetPriors() const { return m_priors; }
			// classes x features: the mean and the variance of every feature in every class.
			const CMatrix& GetMeans() const { return m_means; }
			const CMatrix& GetVariances() const { return m_variances; }

		private:
			CMatrix JointLogLikelihood(const CMatrix& x) const;

		private:
			double m_dVarianceSmoothing;
			CVector m_priors;
			CMatrix m_means;
			CMatrix m_variances;
			bool m_bFitted = false;
		};
	}
}
