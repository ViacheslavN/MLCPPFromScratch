#include "LinearRegression.h"
#include "ModelUtils.h"

#include <cmath>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"
#include "NumericsCore/Decompositions.h"

namespace ML
{
	namespace supervised
	{
		namespace
		{
			// 'x' and 'y' minus their means: a model fitted on centered data needs no
			// intercept, and the penalized models must not penalize the intercept.
			struct SCentered
			{
				SCentered(const CMatrix& x, const CVector& y) : matrix(x), targets(y), means(x.Mean(CMatrix::axisRows)), targetMean(y.Mean())
				{
					for (size_t i = 0; i < matrix.Rows(); ++i)
					{
						double* pRow = matrix.Data() + i * matrix.Cols();
						for (size_t j = 0; j < matrix.Cols(); ++j)
							pRow[j] -= means[j];

						targets[i] -= targetMean;
					}
				}

				// The intercept that goes with 'weights' on the original data.
				double Intercept(const CVector& weights) const { return targetMean - CVector::Dot(means, weights); }

				CMatrix matrix;
				CVector targets;
				CVector means;
				double targetMean;
			};
		}

		// ---- CLinearModel ----

		CVector CLinearModel::Predict(const CMatrix& x) const
		{
			detail::CheckPredict(x, m_weights.Size(), m_bFitted, "Linear model");

			CVector result = CMatrix::Multiply(x, m_weights);
			for (size_t i = 0; i < result.Size(); ++i)
				result[i] += m_dIntercept;

			return result;
		}

		// ---- CLinearRegression ----

		CLinearRegression::CLinearRegression(const SLinearRegressionOptions& options) : m_options(options)
		{}

		void CLinearRegression::Fit(const CMatrix& x, const CVector& y)
		{
			detail::CheckFit(x, y.Size(), "Linear regression");

			m_bFitted = false;
			m_nIterations = 0;
			m_bConverged = false;

			if (m_options.solver == SLinearRegressionOptions::solverNormalEquation)
				FitNormalEquation(x, y);
			else
				FitGradientDescent(x, y);

			m_bFitted = true;
		}

		void CLinearRegression::FitNormalEquation(const CMatrix& x, const CVector& y)
		{
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();

			// A column of ones in front of 'x': its weight is the intercept.
			CMatrix design(nRows, nCols + 1, x.GetAlloc());
			for (size_t i = 0; i < nRows; ++i)
			{
				design(i, 0) = 1;
				for (size_t j = 0; j < nCols; ++j)
					design(i, j + 1) = x(i, j);
			}

			if (nRows < nCols + 1)
				throw CommonLib::CExcBase("Linear regression: {0} samples are not enough for {1} features", nRows, nCols);

			// QR gives the same solution as (X^T X)^-1 X^T y without squaring the
			// condition number of X.
			const CVector solution = numerics::CDecompositions::LeastSquares(design, y);

			m_dIntercept = solution[0];
			m_weights = CVector(nCols, x.GetAlloc());
			for (size_t j = 0; j < nCols; ++j)
				m_weights[j] = solution[j + 1];

			m_bConverged = true;
		}

		void CLinearRegression::FitGradientDescent(const CMatrix& x, const CVector& y)
		{
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();

			m_weights = CVector(nCols, x.GetAlloc());
			m_dIntercept = 0;

			CVector gradient(nCols, x.GetAlloc());
			const double scale = 2.0 / double(nRows);

			for (m_nIterations = 0; m_nIterations < m_options.maxIterations; ++m_nIterations)
			{
				// Gradient of the mean squared error: (2 / n) * X^T * (prediction - y).
				gradient.Fill(0);
				double gradientIntercept = 0;

				for (size_t i = 0; i < nRows; ++i)
				{
					const double* pRow = x.Data() + i * nCols;

					double error = m_dIntercept - y[i];
					for (size_t j = 0; j < nCols; ++j)
						error += pRow[j] * m_weights[j];

					for (size_t j = 0; j < nCols; ++j)
						gradient[j] += error * pRow[j];

					gradientIntercept += error;
				}

				gradientIntercept *= scale;
				double maxGradient = std::fabs(gradientIntercept);
				for (size_t j = 0; j < nCols; ++j)
				{
					gradient[j] *= scale;
					maxGradient = (std::max)(maxGradient, std::fabs(gradient[j]));
				}

				if (!std::isfinite(maxGradient))
				{
					throw CommonLib::CExcBase("Linear regression: gradient descent diverged at iteration {0}; "
						"lower the learning rate or scale the features", m_nIterations);
				}

				if (maxGradient < m_options.tolerance)
				{
					m_bConverged = true;
					break;
				}

				for (size_t j = 0; j < nCols; ++j)
					m_weights[j] -= m_options.learningRate * gradient[j];

				m_dIntercept -= m_options.learningRate * gradientIntercept;
			}
		}

		// ---- CRidgeRegression ----

		CRidgeRegression::CRidgeRegression(double alpha) : m_dAlpha(alpha)
		{
			if (!(alpha >= 0))
				throw CommonLib::CExcBase("Ridge regression: alpha {0} must not be negative", alpha);
		}

		void CRidgeRegression::Fit(const CMatrix& x, const CVector& y)
		{
			detail::CheckFit(x, y.Size(), "Ridge regression");

			m_bFitted = false;

			const SCentered centered(x, y);
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();

			// (Xc^T * Xc + alpha * I) * weights = Xc^T * yc.
			CMatrix gram(nCols, nCols, x.GetAlloc());
			CVector rhs(nCols, x.GetAlloc());
			for (size_t i = 0; i < nRows; ++i)
			{
				const double* pRow = centered.matrix.Data() + i * nCols;
				for (size_t j = 0; j < nCols; ++j)
				{
					for (size_t k = 0; k <= j; ++k)
						gram(j, k) += pRow[j] * pRow[k];	// the lower triangle is all Cholesky reads

					rhs[j] += pRow[j] * centered.targets[i];
				}
			}

			for (size_t j = 0; j < nCols; ++j)
				gram(j, j) += m_dAlpha;

			try
			{
				m_weights = numerics::CDecompositions::SolveCholesky(numerics::CDecompositions::Cholesky(gram), rhs);
			}
			catch (CommonLib::CExcBase& exc)
			{
				exc.AddMsg("Ridge regression: the features are linearly dependent; use alpha > 0");
				throw;
			}

			m_dIntercept = centered.Intercept(m_weights);
			m_bFitted = true;
		}

		// ---- CLassoRegression ----

		CLassoRegression::CLassoRegression(double alpha, size_t maxIterations, double tolerance) : m_dAlpha(alpha),
			m_nMaxIterations(maxIterations), m_dTolerance(tolerance)
		{
			if (!(alpha >= 0))
				throw CommonLib::CExcBase("Lasso regression: alpha {0} must not be negative", alpha);
		}

		void CLassoRegression::Fit(const CMatrix& x, const CVector& y)
		{
			detail::CheckFit(x, y.Size(), "Lasso regression");

			m_bFitted = false;
			m_bConverged = false;

			const SCentered centered(x, y);
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();
			const double* pData = centered.matrix.Data();
			const double count = double(nRows);

			// Mean square of every column.
			CVector squares(nCols, x.GetAlloc());
			for (size_t i = 0; i < nRows; ++i)
			{
				for (size_t j = 0; j < nCols; ++j)
					squares[j] += pData[i * nCols + j] * pData[i * nCols + j] / count;
			}

			m_weights = CVector(nCols, x.GetAlloc());
			CVector residual(centered.targets);		// yc - Xc * weights, kept up to date

			// Coordinate descent: minimize over one weight at a time, the others fixed.
			// For one weight the minimum has a closed form (soft thresholding).
			for (m_nIterations = 0; m_nIterations < m_nMaxIterations && !m_bConverged; ++m_nIterations)
			{
				double maxChange = 0;

				for (size_t j = 0; j < nCols; ++j)
				{
					if (squares[j] == 0)
						continue;	// a constant feature: its weight stays 0

					// Correlation of feature j with the residual that ignores feature j.
					double rho = 0;
					for (size_t i = 0; i < nRows; ++i)
						rho += pData[i * nCols + j] * residual[i];

					rho = rho / count + squares[j] * m_weights[j];

					double weight = 0;
					if (rho > m_dAlpha)
						weight = (rho - m_dAlpha) / squares[j];
					else if (rho < -m_dAlpha)
						weight = (rho + m_dAlpha) / squares[j];

					const double change = weight - m_weights[j];
					if (change != 0)
					{
						for (size_t i = 0; i < nRows; ++i)
							residual[i] -= change * pData[i * nCols + j];

						m_weights[j] = weight;
						maxChange = (std::max)(maxChange, std::fabs(change));
					}
				}

				if (!(maxChange > m_dTolerance))
					m_bConverged = true;
			}

			m_dIntercept = centered.Intercept(m_weights);
			m_bFitted = true;
		}
	}
}
