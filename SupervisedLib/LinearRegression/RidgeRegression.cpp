#include "RidgeRegression.h"
#include "CenteredData.h"
#include "../ModelUtils.h"

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"
#include "NumericsCoreLib/Decompositions.h"

namespace ML
{
	namespace supervised
	{
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

			const detail::TCentered<T> centered(x, y);
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

		// The element types the library supports.
		template class TRidgeRegression<float>;
		template class TRidgeRegression<double>;
	}
}
