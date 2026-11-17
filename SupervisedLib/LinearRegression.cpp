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
			template<class T>
			struct TCentered
			{
				TCentered(const TMatrix<T>& x, const TVector<T>& y) : matrix(x), targets(y), means(x.Mean(TMatrix<T>::axisRows)), targetMean(y.Mean())
				{
					for (size_t i = 0; i < matrix.Rows(); ++i)
					{
						T* pRow = matrix.Data() + i * matrix.Cols();
						for (size_t j = 0; j < matrix.Cols(); ++j)
							pRow[j] -= means[j];

						targets[i] -= targetMean;
					}
				}

				// The intercept that goes with 'weights' on the original data.
				T Intercept(const TVector<T>& weights) const { return targetMean - TVector<T>::Dot(means, weights); }

				TMatrix<T> matrix;
				TVector<T> targets;
				TVector<T> means;
				T targetMean;
			};
		}

		// ---- TLinearModel ----

		template<class T>
		TVector<T> TLinearModel<T>::Predict(const TMatrix<T>& x) const
		{
			detail::CheckPredict(x.Cols(), this->m_weights.Size(), this->m_bFitted, "Linear model");

			TVector<T> result = TMatrix<T>::Multiply(x, this->m_weights);
			for (size_t i = 0; i < result.Size(); ++i)
				result[i] += this->m_intercept;

			return result;
		}

		// ---- TLinearRegression ----

		template<class T>
		TLinearRegression<T>::TLinearRegression(const SLinearRegressionOptions& options) : m_options(options)
		{}

		template<class T>
		void TLinearRegression<T>::Fit(const TMatrix<T>& x, const TVector<T>& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Linear regression");

			this->m_bFitted = false;
			m_nIterations = 0;
			m_bConverged = false;

			if (m_options.solver == SLinearRegressionOptions::solverNormalEquation)
				FitNormalEquation(x, y);
			else
				FitGradientDescent(x, y);

			this->m_bFitted = true;
		}

		template<class T>
		void TLinearRegression<T>::FitNormalEquation(const TMatrix<T>& x, const TVector<T>& y)
		{
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();

			// A column of ones in front of 'x': its weight is the intercept.
			TMatrix<T> design(nRows, nCols + 1, x.GetAlloc());
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
			const TVector<T> solution = numerics::TDecompositions<T>::LeastSquares(design, y);

			this->m_intercept = solution[0];
			this->m_weights = TVector<T>(nCols, x.GetAlloc());
			for (size_t j = 0; j < nCols; ++j)
				this->m_weights[j] = solution[j + 1];

			m_bConverged = true;
		}

		template<class T>
		void TLinearRegression<T>::FitGradientDescent(const TMatrix<T>& x, const TVector<T>& y)
		{
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();

			this->m_weights = TVector<T>(nCols, x.GetAlloc());
			this->m_intercept = 0;

			TVector<T> gradient(nCols, x.GetAlloc());
			const T scale = T(2) / T(nRows);
			const T learningRate = T(m_options.learningRate);
			const T tolerance = T(m_options.tolerance);

			for (m_nIterations = 0; m_nIterations < m_options.maxIterations; ++m_nIterations)
			{
				// Gradient of the mean squared error: (2 / n) * X^T * (prediction - y).
				gradient.Fill(0);
				T gradientIntercept = 0;

				for (size_t i = 0; i < nRows; ++i)
				{
					const T* pRow = x.Data() + i * nCols;

					T error = this->m_intercept - y[i];
					for (size_t j = 0; j < nCols; ++j)
						error += pRow[j] * this->m_weights[j];

					for (size_t j = 0; j < nCols; ++j)
						gradient[j] += error * pRow[j];

					gradientIntercept += error;
				}

				gradientIntercept *= scale;
				T maxGradient = std::fabs(gradientIntercept);
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

				if (maxGradient < tolerance)
				{
					m_bConverged = true;
					break;
				}

				for (size_t j = 0; j < nCols; ++j)
					this->m_weights[j] -= learningRate * gradient[j];

				this->m_intercept -= learningRate * gradientIntercept;
			}
		}

		// ---- TRidgeRegression ----

		template<class T>
		TRidgeRegression<T>::TRidgeRegression(T alpha) : m_alpha(alpha)
		{
			if (!(alpha >= 0))
				throw CommonLib::CExcBase("Ridge regression: alpha {0} must not be negative", alpha);
		}

		template<class T>
		void TRidgeRegression<T>::Fit(const TMatrix<T>& x, const TVector<T>& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Ridge regression");

			this->m_bFitted = false;

			const TCentered<T> centered(x, y);
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();

			// (Xc^T * Xc + alpha * I) * weights = Xc^T * yc.
			TMatrix<T> gram(nCols, nCols, x.GetAlloc());
			TVector<T> rhs(nCols, x.GetAlloc());
			for (size_t i = 0; i < nRows; ++i)
			{
				const T* pRow = centered.matrix.Data() + i * nCols;
				for (size_t j = 0; j < nCols; ++j)
				{
					for (size_t k = 0; k <= j; ++k)
						gram(j, k) += pRow[j] * pRow[k];	// the lower triangle is all Cholesky reads

					rhs[j] += pRow[j] * centered.targets[i];
				}
			}

			for (size_t j = 0; j < nCols; ++j)
				gram(j, j) += m_alpha;

			try
			{
				this->m_weights = numerics::TDecompositions<T>::SolveCholesky(numerics::TDecompositions<T>::Cholesky(gram), rhs);
			}
			catch (CommonLib::CExcBase& exc)
			{
				exc.AddMsg("Ridge regression: the features are linearly dependent; use alpha > 0");
				throw;
			}

			this->m_intercept = centered.Intercept(this->m_weights);
			this->m_bFitted = true;
		}

		// ---- TLassoRegression ----

		template<class T>
		TLassoRegression<T>::TLassoRegression(T alpha, size_t maxIterations, T tolerance) : m_alpha(alpha),
			m_nMaxIterations(maxIterations), m_tolerance(tolerance)
		{
			if (!(alpha >= 0))
				throw CommonLib::CExcBase("Lasso regression: alpha {0} must not be negative", alpha);
		}

		template<class T>
		void TLassoRegression<T>::Fit(const TMatrix<T>& x, const TVector<T>& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Lasso regression");

			this->m_bFitted = false;
			m_bConverged = false;

			const TCentered<T> centered(x, y);
			const size_t nRows = x.Rows();
			const size_t nCols = x.Cols();
			const T* pData = centered.matrix.Data();
			const T count = T(nRows);

			// Mean square of every column.
			TVector<T> squares(nCols, x.GetAlloc());
			for (size_t i = 0; i < nRows; ++i)
			{
				for (size_t j = 0; j < nCols; ++j)
					squares[j] += pData[i * nCols + j] * pData[i * nCols + j] / count;
			}

			this->m_weights = TVector<T>(nCols, x.GetAlloc());
			TVector<T> residual(centered.targets);		// yc - Xc * weights, kept up to date

			// Coordinate descent: minimize over one weight at a time, the others fixed.
			// For one weight the minimum has a closed form (soft thresholding).
			for (m_nIterations = 0; m_nIterations < m_nMaxIterations && !m_bConverged; ++m_nIterations)
			{
				T maxChange = 0;

				for (size_t j = 0; j < nCols; ++j)
				{
					if (squares[j] == 0)
						continue;	// a constant feature: its weight stays 0

					// Correlation of feature j with the residual that ignores feature j.
					T rho = 0;
					for (size_t i = 0; i < nRows; ++i)
						rho += pData[i * nCols + j] * residual[i];

					rho = rho / count + squares[j] * this->m_weights[j];

					T weight = 0;
					if (rho > m_alpha)
						weight = (rho - m_alpha) / squares[j];
					else if (rho < -m_alpha)
						weight = (rho + m_alpha) / squares[j];

					const T change = weight - this->m_weights[j];
					if (change != 0)
					{
						for (size_t i = 0; i < nRows; ++i)
							residual[i] -= change * pData[i * nCols + j];

						this->m_weights[j] = weight;
						maxChange = (std::max)(maxChange, std::fabs(change));
					}
				}

				if (!(maxChange > m_tolerance))
					m_bConverged = true;
			}

			this->m_intercept = centered.Intercept(this->m_weights);
			this->m_bFitted = true;
		}

		// The element types the library supports.
		template class TLinearModel<float>;
		template class TLinearModel<double>;
		template class TLinearRegression<float>;
		template class TLinearRegression<double>;
		template class TRidgeRegression<float>;
		template class TRidgeRegression<double>;
		template class TLassoRegression<float>;
		template class TLassoRegression<double>;
	}
}