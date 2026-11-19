#include "StandardScaler.h"

#include <cmath>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"
#include "NumericsCoreLib/Statistics/Statistics.h"

namespace ML
{
	namespace data
	{
		using numerics::TMatrix;
		using numerics::TVector;

		template<class T>
		void TStandardScaler<T>::Fit(const TMatrix<T>& x)
		{
			if (x.Rows() == 0)
				throw CommonLib::CExcBase("Standard scaler: no data to fit, size {0} x {1}", x.Rows(), x.Cols());

			m_bFitted = false;

			m_means = x.Mean(TMatrix<T>::axisRows);
			m_scales = numerics::TStatistics<T>::Variance(x, TMatrix<T>::axisRows, numerics::vtPopulation);

			for (size_t j = 0; j < m_scales.Size(); ++j)
			{
				const T deviation = std::sqrt(m_scales[j]);

				// A constant column has nothing to scale: dividing by 1 leaves it at 0.
				m_scales[j] = deviation > 0 ? deviation : T(1);
			}

			m_bFitted = true;
		}

		template<class T>
		TMatrix<T> TStandardScaler<T>::Transform(const TMatrix<T>& x) const
		{
			Check(x);

			const size_t nCols = x.Cols();

			TMatrix<T> result(x);
			for (size_t i = 0; i < x.Rows(); ++i)
			{
				T* pRow = result.Data() + i * nCols;
				for (size_t j = 0; j < nCols; ++j)
					pRow[j] = (pRow[j] - m_means[j]) / m_scales[j];
			}

			return result;
		}

		template<class T>
		TMatrix<T> TStandardScaler<T>::FitTransform(const TMatrix<T>& x)
		{
			Fit(x);
			return Transform(x);
		}

		template<class T>
		TMatrix<T> TStandardScaler<T>::InverseTransform(const TMatrix<T>& x) const
		{
			Check(x);

			const size_t nCols = x.Cols();

			TMatrix<T> result(x);
			for (size_t i = 0; i < x.Rows(); ++i)
			{
				T* pRow = result.Data() + i * nCols;
				for (size_t j = 0; j < nCols; ++j)
					pRow[j] = pRow[j] * m_scales[j] + m_means[j];
			}

			return result;
		}

		template<class T>
		void TStandardScaler<T>::Check(const TMatrix<T>& x) const
		{
			if (!m_bFitted)
				throw CommonLib::CExcBase("Standard scaler: the scaler is not fitted");

			if (x.Cols() != m_means.Size())
				throw CommonLib::CExcBase("Standard scaler: the data has {0} columns, the scaler was fitted with {1}", x.Cols(), m_means.Size());
		}

		// The element types the library supports.
		template class TStandardScaler<float>;
		template class TStandardScaler<double>;
	}
}
