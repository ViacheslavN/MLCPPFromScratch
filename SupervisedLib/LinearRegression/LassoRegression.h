#pragma once

#include "LinearRegression.h"

namespace ML
{
	namespace supervised
	{
		// Least squares with an L1 penalty: minimizes
		//   sum of (prediction - y)^2 / (2 * n) + alpha * sum of |weights|
		// for n samples. The penalty sets the weights of unhelpful features to exactly
		// zero. The intercept is not penalized. Solved by coordinate descent.
		template<class T>
		class TLassoRegression : public TLinearModel<T>
		{
		public:
			// 'alpha' >= 0. Stops when no weight changes by more than 'tolerance' in a
			// pass over the features, or after 'maxIterations' passes. The default
			// tolerance is 1e-8 for double and 1e-5 for float, whose rounding is coarser.
			explicit TLassoRegression(T alpha = T(1), size_t maxIterations = 1000, T tolerance = T(sizeof(T) == sizeof(float) ? 1e-5 : 1e-8));

			void Fit(const TMatrix<T>& x, const TVector<T>& y) override;
			using IRegressorT<T>::Fit;		// the overload for a vector of one feature

			size_t GetIterations() const { return m_nIterations; }
			bool IsConverged() const { return m_bConverged; }

		private:
			T m_alpha;
			size_t m_nMaxIterations;
			T m_tolerance;
			size_t m_nIterations = 0;
			bool m_bConverged = false;
		};

		typedef TLassoRegression<double> CLassoRegression;
		typedef TLassoRegression<float> CLassoRegressionF;
	}
}
