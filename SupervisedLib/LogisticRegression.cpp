#include "LogisticRegression.h"
#include "ModelUtils.h"

#include <cmath>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"
#include "NumericsCoreLib/Statistics/Statistics.h"

namespace ML
{
	namespace supervised
	{
		CLogisticRegression::CLogisticRegression(const SLogisticRegressionOptions& options) : m_options(options)
		{
			if (!(options.l2 >= 0))
				throw CommonLib::CExcBase("Logistic regression: l2 {0} must not be negative", options.l2);
		}

		void CLogisticRegression::Fit(const CMatrix& x, const CVectorI& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Logistic regression");

			const size_t nClasses = detail::CountClasses(y, "Logistic regression");
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();
			const double count = double(nRows);

			m_bFitted = false;
			m_bConverged = false;
			m_weights = CMatrix(nClasses, nCols, x.GetAlloc());
			m_intercepts = CVector(nClasses, x.GetAlloc());

			CMatrix gradient(nClasses, nCols, x.GetAlloc());
			CVector gradientIntercepts(nClasses, x.GetAlloc());

			for (m_nIterations = 0; m_nIterations < m_options.maxIterations; ++m_nIterations)
			{
				// Gradient of the mean cross-entropy with respect to the scores:
				// (probability - 1) for the true class of a sample, probability for the others.
				CMatrix errors = numerics::CStatistics::Softmax(Scores(x), CMatrix::axisCols);
				for (size_t i = 0; i < nRows; ++i)
					errors(i, size_t(y[i])) -= 1;

				// Chain rule to the weights: errors^T * x / n, plus the penalty.
				gradient.Fill(0);
				gradientIntercepts.Fill(0);
				for (size_t i = 0; i < nRows; ++i)
				{
					const double* pRow = x.Data() + i * nCols;
					for (size_t k = 0; k < nClasses; ++k)
					{
						const double error = errors(i, k);
						double* pGradient = gradient.Data() + k * nCols;
						for (size_t j = 0; j < nCols; ++j)
							pGradient[j] += error * pRow[j];

						gradientIntercepts[k] += error;
					}
				}

				double maxGradient = 0;
				for (size_t k = 0; k < nClasses; ++k)
				{
					for (size_t j = 0; j < nCols; ++j)
					{
						gradient(k, j) = gradient(k, j) / count + m_options.l2 * m_weights(k, j);
						maxGradient = (std::max)(maxGradient, std::fabs(gradient(k, j)));
					}

					gradientIntercepts[k] /= count;
					maxGradient = (std::max)(maxGradient, std::fabs(gradientIntercepts[k]));
				}

				if (!std::isfinite(maxGradient))
				{
					throw CommonLib::CExcBase("Logistic regression: gradient descent diverged at iteration {0}; "
						"lower the learning rate or scale the features", m_nIterations);
				}

				if (maxGradient < m_options.tolerance)
				{
					m_bConverged = true;
					break;
				}

				gradient *= m_options.learningRate;
				gradientIntercepts *= m_options.learningRate;
				m_weights -= gradient;
				m_intercepts -= gradientIntercepts;
			}

			m_bFitted = true;
		}

		CVectorI CLogisticRegression::Predict(const CMatrix& x) const
		{
			detail::CheckPredict(x.Cols(), m_weights.Cols(), m_bFitted, "Logistic regression");

			// The largest score is also the largest probability.
			const CMatrix scores = Scores(x);

			CVectorI result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
				result[i] = int(detail::ArgMax(scores.Data() + i * scores.Cols(), scores.Cols()));

			return result;
		}

		CMatrix CLogisticRegression::PredictProbability(const CMatrix& x) const
		{
			detail::CheckPredict(x.Cols(), m_weights.Cols(), m_bFitted, "Logistic regression");

			return numerics::CStatistics::Softmax(Scores(x), CMatrix::axisCols);
		}

		// Rows() x classes: x * weights^T + intercepts.
		CMatrix CLogisticRegression::Scores(const CMatrix& x) const
		{
			const size_t nClasses = m_weights.Rows();
			const size_t nCols = x.Cols();

			CMatrix scores(x.Rows(), nClasses, x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
			{
				const double* pRow = x.Data() + i * nCols;
				for (size_t k = 0; k < nClasses; ++k)
				{
					const double* pWeights = m_weights.Data() + k * nCols;

					double sum = m_intercepts[k];
					for (size_t j = 0; j < nCols; ++j)
						sum += pRow[j] * pWeights[j];

					scores(i, k) = sum;
				}
			}

			return scores;
		}
	}
}
