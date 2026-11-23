#include "LinearRegression.h"
#include "../ModelUtils.h"

#include <cmath>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"
#include "NumericsCoreLib/Decompositions.h"

namespace ML
{
	namespace supervised
	{
		template<class T>
		TVector<T> TLinearModel<T>::Predict(const TMatrix<T>& x) const
		{
			detail::CheckPredict(x.Cols(), this->m_weights.Size(), this->m_bFitted, "Linear model");

			TVector<T> result = TMatrix<T>::Multiply(x, this->m_weights);
			for (size_t i = 0; i < result.Size(); ++i)
				result[i] += this->m_intercept;

			return result;
		}

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

		// The element types the library supports.
		template class TLinearModel<float>;
		template class TLinearModel<double>;
		template class TLinearRegression<float>;
		template class TLinearRegression<double>;
	}
}
