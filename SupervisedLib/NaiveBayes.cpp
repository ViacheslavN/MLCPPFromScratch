#include "NaiveBayes.h"
#include "ModelUtils.h"

#include <cmath>
#include <limits>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"
#include "NumericsCore/Statistics/Statistics.h"

namespace ML
{
	namespace supervised
	{
		CGaussianNaiveBayes::CGaussianNaiveBayes(double varianceSmoothing) : m_dVarianceSmoothing(varianceSmoothing)
		{
			if (!(varianceSmoothing >= 0))
				throw CommonLib::CExcBase("Naive Bayes: the variance smoothing {0} must not be negative", varianceSmoothing);
		}

		void CGaussianNaiveBayes::Fit(const CMatrix& x, const CVectorI& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Naive Bayes");

			const size_t nClasses = detail::CountClasses(y, "Naive Bayes");
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();

			m_bFitted = false;
			m_priors = CVector(nClasses, x.GetAlloc());
			m_means = CMatrix(nClasses, nCols, x.GetAlloc());
			m_variances = CMatrix(nClasses, nCols, x.GetAlloc());

			// Samples and feature sums per class, then the means.
			for (size_t i = 0; i < nRows; ++i)
			{
				const size_t nClass = size_t(y[i]);
				m_priors[nClass] += 1;
				for (size_t j = 0; j < nCols; ++j)
					m_means(nClass, j) += x(i, j);
			}

			for (size_t k = 0; k < nClasses; ++k)
			{
				for (size_t j = 0; j < nCols && m_priors[k] > 0; ++j)
					m_means(k, j) /= m_priors[k];
			}

			// Squared deviations from the class mean, then the variances.
			for (size_t i = 0; i < nRows; ++i)
			{
				const size_t nClass = size_t(y[i]);
				for (size_t j = 0; j < nCols; ++j)
				{
					const double deviation = x(i, j) - m_means(nClass, j);
					m_variances(nClass, j) += deviation * deviation;
				}
			}

			double smoothing = m_dVarianceSmoothing * numerics::CStatistics::Variance(x, CMatrix::axisRows, numerics::vtPopulation).Max();
			if (smoothing == 0)
				smoothing = std::numeric_limits<double>::min();	// all features constant: avoid 0 / 0

			for (size_t k = 0; k < nClasses; ++k)
			{
				for (size_t j = 0; j < nCols; ++j)
				{
					if (m_priors[k] > 0)
						m_variances(k, j) /= m_priors[k];

					m_variances(k, j) += smoothing;
				}

				m_priors[k] /= double(nRows);
			}

			m_bFitted = true;
		}

		CVectorI CGaussianNaiveBayes::Predict(const CMatrix& x) const
		{
			detail::CheckPredict(x.Cols(), m_means.Cols(), m_bFitted, "Naive Bayes");

			const CMatrix likelihood = JointLogLikelihood(x);

			CVectorI result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
				result[i] = int(detail::ArgMax(likelihood.Data() + i * likelihood.Cols(), likelihood.Cols()));

			return result;
		}

		CMatrix CGaussianNaiveBayes::PredictProbability(const CMatrix& x) const
		{
			detail::CheckPredict(x.Cols(), m_means.Cols(), m_bFitted, "Naive Bayes");

			// Bayes' rule: P(class | row) = exp(joint) / sum over the classes of exp(joint),
			// which is the softmax of the joint log-likelihoods.
			return numerics::CStatistics::Softmax(JointLogLikelihood(x), CMatrix::axisCols);
		}

		// Rows() x classes: log(prior) + sum over the features of log(normal density).
		// Logarithms, because the product of many small densities underflows to 0.
		CMatrix CGaussianNaiveBayes::JointLogLikelihood(const CMatrix& x) const
		{
			const size_t nClasses = m_means.Rows();
			const size_t nCols = x.Cols();
			const double log2Pi = std::log(2 * 3.14159265358979323846);

			CMatrix result(x.Rows(), nClasses, x.GetAlloc());
			for (size_t k = 0; k < nClasses; ++k)
			{
				// A class number with no training samples can never be predicted.
				if (m_priors[k] == 0)
				{
					for (size_t i = 0; i < x.Rows(); ++i)
						result(i, k) = -std::numeric_limits<double>::infinity();

					continue;
				}

				// The part that does not depend on the row.
				double constant = std::log(m_priors[k]);
				for (size_t j = 0; j < nCols; ++j)
					constant -= 0.5 * (log2Pi + std::log(m_variances(k, j)));

				for (size_t i = 0; i < x.Rows(); ++i)
				{
					double sum = constant;
					for (size_t j = 0; j < nCols; ++j)
					{
						const double deviation = x(i, j) - m_means(k, j);
						sum -= 0.5 * deviation * deviation / m_variances(k, j);
					}

					result(i, k) = sum;
				}
			}

			return result;
		}
	}
}
