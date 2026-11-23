#include "LassoRegression.h"
#include "CenteredData.h"
#include "../ModelUtils.h"

#include <cmath>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace supervised
	{
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

			const detail::TCentered<T> centered(x, y);
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
		template class TLassoRegression<float>;
		template class TLassoRegression<double>;
	}
}
