#pragma once

#include <cmath>

#include "MetricsUtils.h"

namespace ML
{
	namespace evaluation
	{
		// How close predicted numbers are to the true ones. Static methods only.
		//
		// The methods are templates: 'yTrue' and 'yPredicted' are vectors of any element
		// type (float, double, int), deduced from the arguments. The sums are
		// accumulated in double and the result is a double, whatever the element type.
		// Vectors of different sizes and empty vectors throw CommonLib::CExcBase.
		class CRegressionMetrics
		{
		public:
			// Mean squared error: the mean of (true - predicted)^2. Large errors count
			// much more than small ones. In the squared unit of the target.
			template<class T>
			static double MSE(const numerics::TVector<T>& yTrue, const numerics::TVector<T>& yPredicted)
			{
				detail::CheckPair(yTrue.Size(), yPredicted.Size(), "MSE");

				double sum = 0;
				for (size_t i = 0; i < yTrue.Size(); ++i)
				{
					const double error = double(yTrue[i]) - double(yPredicted[i]);
					sum += error * error;
				}

				return sum / double(yTrue.Size());
			}

			// Root mean squared error: sqrt(MSE). In the unit of the target.
			template<class T>
			static double RMSE(const numerics::TVector<T>& yTrue, const numerics::TVector<T>& yPredicted)
			{
				return std::sqrt(MSE(yTrue, yPredicted));
			}

			// Mean absolute error: the mean of |true - predicted|. In the unit of the
			// target; less sensitive to a few large errors than RMSE.
			template<class T>
			static double MAE(const numerics::TVector<T>& yTrue, const numerics::TVector<T>& yPredicted)
			{
				detail::CheckPair(yTrue.Size(), yPredicted.Size(), "MAE");

				double sum = 0;
				for (size_t i = 0; i < yTrue.Size(); ++i)
					sum += std::fabs(double(yTrue[i]) - double(yPredicted[i]));

				return sum / double(yTrue.Size());
			}

			// R^2, the coefficient of determination:
			//   1 - sum of (true - predicted)^2 / sum of (true - mean of true)^2.
			// 1 is a perfect fit, 0 is no better than always predicting the mean, and it
			// is negative for predictions worse than that. When all true values are
			// equal the ratio is 0 / 0: the result is 1 for exact predictions, else 0.
			template<class T>
			static double R2(const numerics::TVector<T>& yTrue, const numerics::TVector<T>& yPredicted)
			{
				detail::CheckPair(yTrue.Size(), yPredicted.Size(), "R2");

				double mean = 0;
				for (size_t i = 0; i < yTrue.Size(); ++i)
					mean += double(yTrue[i]);

				mean /= double(yTrue.Size());

				double residual = 0;
				double total = 0;
				for (size_t i = 0; i < yTrue.Size(); ++i)
				{
					const double error = double(yTrue[i]) - double(yPredicted[i]);
					residual += error * error;
					total += (double(yTrue[i]) - mean) * (double(yTrue[i]) - mean);
				}

				if (total == 0)
					return residual == 0 ? 1.0 : 0.0;

				return 1 - residual / total;
			}
		};
	}
}
