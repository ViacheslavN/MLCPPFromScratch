#include "SVM.h"
#include "ModelUtils.h"

#include <cmath>
#include <limits>
#include <vector>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"
#include "NumericsCore/Statistics/Statistics.h"

namespace ML
{
	namespace supervised
	{
		CSVMClassifier::CSVMClassifier(const SSVMOptions& options) : m_options(options)
		{
			if (!(options.c > 0))
				throw CommonLib::CExcBase("SVM: c {0} must be positive", options.c);

			if (!(options.gamma >= 0))
				throw CommonLib::CExcBase("SVM: gamma {0} must not be negative", options.gamma);
		}

		void CSVMClassifier::Fit(const CMatrix& x, const CVectorI& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "SVM");

			if (detail::CountClasses(y, "SVM") != 2)
				throw CommonLib::CExcBase("SVM: only two classes are supported, with the labels 0 and 1");

			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();
			const double c = m_options.c;

			m_bFitted = false;
			m_bConverged = false;
			m_nFeatures = nCols;

			m_dGamma = m_options.gamma;
			if (m_options.kernel == SSVMOptions::kernelRbf && m_dGamma == 0)
			{
				// Variance of all the values of 'x' taken together.
				const double mean = x.Mean();
				double variance = 0;
				for (size_t i = 0; i < x.Size(); ++i)
					variance += (x.Data()[i] - mean) * (x.Data()[i] - mean);

				variance /= double(x.Size());
				m_dGamma = variance > 0 ? 1.0 / (double(nCols) * variance) : 1.0;
			}

			// The class signs: +1 for class 1, -1 for class 0.
			std::vector<double> sign(nRows);
			for (size_t i = 0; i < nRows; ++i)
				sign[i] = y[i] == 1 ? 1.0 : -1.0;

			// The kernel between all pairs of samples.
			CMatrix kernel(nRows, nRows, x.GetAlloc());
			for (size_t i = 0; i < nRows; ++i)
			{
				for (size_t j = i; j < nRows; ++j)
				{
					kernel(i, j) = Kernel(x.Data() + i * nCols, x.Data() + j * nCols, nCols);
					kernel(j, i) = kernel(i, j);
				}
			}

			// The dual problem: maximize  sum(alpha) - 1/2 * sum over i, j of
			//   alpha[i] * alpha[j] * sign[i] * sign[j] * K(i, j)
			// with 0 <= alpha[i] <= c and sum(alpha[i] * sign[i]) = 0.
			//
			// error[i] = f(i) - sign[i], where f(i) = sum over k of
			// alpha[k] * sign[k] * K(k, i) is the score of sample i without the
			// intercept. All alphas start at 0, so f = 0.
			std::vector<double> alpha(nRows, 0.0);
			std::vector<double> error(nRows);
			for (size_t i = 0; i < nRows; ++i)
				error[i] = -sign[i];

			double upValue = 0;
			double lowValue = 0;

			for (m_nIterations = 0; m_nIterations < m_options.maxIterations; ++m_nIterations)
			{
				// The pair that violates the optimality conditions the most. "Up" are
				// the samples whose alpha * sign can still grow, "low" the ones where it
				// can still shrink. At the optimum -error of every up sample is below
				// -error of every low sample (and both meet at the intercept).
				size_t nUp = nRows;
				size_t nLow = nRows;
				upValue = -std::numeric_limits<double>::infinity();
				lowValue = std::numeric_limits<double>::infinity();

				for (size_t i = 0; i < nRows; ++i)
				{
					const bool bUp = sign[i] > 0 ? alpha[i] < c : alpha[i] > 0;
					const bool bLow = sign[i] > 0 ? alpha[i] > 0 : alpha[i] < c;

					if (bUp && -error[i] > upValue)
					{
						upValue = -error[i];
						nUp = i;
					}

					if (bLow && -error[i] < lowValue)
					{
						lowValue = -error[i];
						nLow = i;
					}
				}

				if (nUp == nRows || nLow == nRows || upValue - lowValue < m_options.tolerance)
				{
					m_bConverged = true;
					break;
				}

				const size_t i = nUp;
				const size_t j = nLow;

				// With the other alphas fixed, the constraint leaves one free variable.
				// Along it the objective is a parabola: 'eta' is its second derivative,
				// and the unclipped optimum for alpha[j] is below. [low, high] is where
				// both alphas stay inside [0, c].
				double low = 0;
				double high = 0;
				if (sign[i] != sign[j])
				{
					low = (std::max)(0.0, alpha[j] - alpha[i]);
					high = (std::min)(c, c + alpha[j] - alpha[i]);
				}
				else
				{
					low = (std::max)(0.0, alpha[i] + alpha[j] - c);
					high = (std::min)(c, alpha[i] + alpha[j]);
				}

				double eta = 2 * kernel(i, j) - kernel(i, i) - kernel(j, j);
				if (!(eta < 0))
					eta = -1e-12;	// two identical samples: keep the step finite

				double alphaJ = alpha[j] - sign[j] * (error[i] - error[j]) / eta;
				alphaJ = (std::min)(high, (std::max)(low, alphaJ));

				double alphaI = alpha[i] + sign[i] * sign[j] * (alpha[j] - alphaJ);

				// Rounding can leave an alpha a hair away from 0 or c. Such a sample
				// would still count as movable and be picked again and again for steps
				// of size ~1e-17, so put it exactly on the bound.
				const auto snap = [c](double value)
				{
					if (value < 1e-12 * c)
						return 0.0;

					return value > c * (1 - 1e-12) ? c : value;
				};

				alphaI = snap(alphaI);
				alphaJ = snap(alphaJ);

				const double changeI = (alphaI - alpha[i]) * sign[i];
				const double changeJ = (alphaJ - alpha[j]) * sign[j];
				for (size_t k = 0; k < nRows; ++k)
					error[k] += changeI * kernel(i, k) + changeJ * kernel(j, k);

				alpha[i] = alphaI;
				alpha[j] = alphaJ;
			}

			// At the optimum -error equals the intercept for every sample strictly
			// inside (0, c); 'upValue' and 'lowValue' bracket it within the tolerance.
			m_dIntercept = std::isfinite(upValue) && std::isfinite(lowValue) ? (upValue + lowValue) / 2 : 0.0;

			// Keep the support vectors only. Alphas this close to 0 are rounding noise.
			std::vector<size_t> support;
			for (size_t i = 0; i < nRows; ++i)
			{
				if (alpha[i] > 1e-12 * c)
					support.push_back(i);
			}

			m_supportVectors = CMatrix(support.size(), nCols, x.GetAlloc());
			m_coefficients = CVector(support.size(), x.GetAlloc());
			for (size_t k = 0; k < support.size(); ++k)
			{
				m_coefficients[k] = alpha[support[k]] * sign[support[k]];
				for (size_t j = 0; j < nCols; ++j)
					m_supportVectors(k, j) = x(support[k], j);
			}

			m_bFitted = true;
		}

		CVectorI CSVMClassifier::Predict(const CMatrix& x) const
		{
			const CVector scores = DecisionFunction(x);

			CVectorI result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
				result[i] = scores[i] >= 0 ? 1 : 0;

			return result;
		}

		CVector CSVMClassifier::DecisionFunction(const CMatrix& x) const
		{
			detail::CheckPredict(x.Cols(), m_nFeatures, m_bFitted, "SVM");

			CVector result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
			{
				const double* pRow = x.Data() + i * m_nFeatures;

				double sum = m_dIntercept;
				for (size_t k = 0; k < m_supportVectors.Rows(); ++k)
					sum += m_coefficients[k] * Kernel(m_supportVectors.Data() + k * m_nFeatures, pRow, m_nFeatures);

				result[i] = sum;
			}

			return result;
		}

		double CSVMClassifier::Kernel(const double* pA, const double* pB, size_t nSize) const
		{
			double sum = 0;

			if (m_options.kernel == SSVMOptions::kernelLinear)
			{
				for (size_t i = 0; i < nSize; ++i)
					sum += pA[i] * pB[i];

				return sum;
			}

			for (size_t i = 0; i < nSize; ++i)
				sum += (pA[i] - pB[i]) * (pA[i] - pB[i]);

			return std::exp(-m_dGamma * sum);
		}
	}
}
