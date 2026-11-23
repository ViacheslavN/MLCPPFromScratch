#pragma once

#include "LinearRegression.h"

namespace ML
{
	namespace supervised
	{
		// Least squares with an L2 penalty: minimizes
		//   sum of (prediction - y)^2 + alpha * sum of weights^2.
		// The penalty shrinks the weights and makes the solution unique even for
		// linearly dependent features. The intercept is not penalized. Solved exactly
		// (Cholesky decomposition).
		template<class T>
		class TRidgeRegression : public TLinearModel<T>
		{
		public:
			// 'alpha' >= 0; 0 is ordinary least squares.
			explicit TRidgeRegression(T alpha = T(1));

			void Fit(const TMatrix<T>& x, const TVector<T>& y) override;
			using IRegressorT<T>::Fit;		// the overload for a vector of one feature

		private:
			T m_alpha;
		};

		typedef TRidgeRegression<double> CRidgeRegression;
		typedef TRidgeRegression<float> CRidgeRegressionF;
	}
}
