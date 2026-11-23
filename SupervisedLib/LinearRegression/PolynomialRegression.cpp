#include "PolynomialRegression.h"
#include "LinearRegression.h"
#include "RidgeRegression.h"
#include "../ModelUtils.h"

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace supervised
	{
		template<class T>
		TPolynomialRegression<T>::TPolynomialRegression(size_t nDegree, T alpha) : m_features(nDegree), m_alpha(alpha)
		{
			if (!(alpha >= 0))
				throw CommonLib::CExcBase("Polynomial regression: alpha {0} must not be negative", alpha);
		}

		template<class T>
		void TPolynomialRegression<T>::Fit(const TMatrix<T>& x, const TVector<T>& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Polynomial regression");

			m_bFitted = false;

			const TMatrix<T> features = m_features.FitTransform(x);

			try
			{
				if (m_alpha > 0)
				{
					TRidgeRegression<T> model(m_alpha);
					model.Fit(features, y);
					m_weights = model.GetWeights();
					m_intercept = model.GetIntercept();
				}
				else
				{
					TLinearRegression<T> model;
					model.Fit(features, y);
					m_weights = model.GetWeights();
					m_intercept = model.GetIntercept();
				}
			}
			catch (CommonLib::CExcBase& exc)
			{
				exc.AddMsgT("Polynomial regression: failed to fit {0} terms of degree {1} on {2} samples",
					m_features.GetTerms().size(), m_features.GetDegree(), x.Rows());
				throw;
			}

			m_bFitted = true;
		}

		template<class T>
		TVector<T> TPolynomialRegression<T>::Predict(const TMatrix<T>& x) const
		{
			if (!m_bFitted)
				throw CommonLib::CExcBase("Polynomial regression: the model is not fitted");

			// Transform checks the number of features.
			TVector<T> result = TMatrix<T>::Multiply(m_features.Transform(x), m_weights);
			for (size_t i = 0; i < result.Size(); ++i)
				result[i] += m_intercept;

			return result;
		}

		// The element types the library supports.
		template class TPolynomialRegression<float>;
		template class TPolynomialRegression<double>;
	}
}
