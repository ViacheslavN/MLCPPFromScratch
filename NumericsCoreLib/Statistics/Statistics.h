#pragma once

#include <cstddef>
#include <type_traits>

#include "../Matrix.h"
#include "../Vector.h"

namespace ML
{
	namespace numerics
	{
		// Which divisor a variance or a covariance uses, for n values.
		enum EVarianceType
		{
			vtSample,		// n - 1: the unbiased estimate from a sample; needs n >= 2
			vtPopulation	// n: the data is the whole population; needs n >= 1
		};

		// Statistics of vectors and matrices. Static methods only. T is float or double
		// (the member functions are compiled for these two in Statistics.cpp).
		//
		// The methods with an axis work like the reductions of TMatrix: axisRows gives one
		// result per column, axisCols one result per row. The results use the allocator of
		// the argument. Too few values and wrong sizes throw CommonLib::CExcBase.
		template<class T>
		class TStatistics
		{
		public:
			typedef T TValue;
			typedef typename TMatrix<T>::EAxis EAxis;

			// ---- Mean ----

			static T Mean(const TVector<T>& vec);
			static TVector<T> Mean(const TMatrix<T>& matrix, EAxis axis);

			// ---- Variance: the mean squared deviation from the mean ----

			static T Variance(const TVector<T>& vec, EVarianceType type = vtSample);
			static TVector<T> Variance(const TMatrix<T>& matrix, EAxis axis, EVarianceType type = vtSample);

			// ---- Covariance matrix ----

			// The rows of 'matrix' are observations, the columns are variables. The
			// result is Cols() x Cols() and symmetric: element (i, j) is the covariance
			// of columns i and j, the diagonal holds the variances of the columns.
			static TMatrix<T> Covariance(const TMatrix<T>& matrix, EVarianceType type = vtSample);

			// ---- Log-sum-exp: log(exp(x[0]) + exp(x[1]) + ...) ----

			// Computed as max + log(sum of exp(x[i] - max)), so exp() cannot overflow for
			// large values or lose everything to underflow for very negative ones.
			// -infinity for no values.
			static T LogSumExp(const TVector<T>& vec);
			static TVector<T> LogSumExp(const TMatrix<T>& matrix, EAxis axis);

			// ---- Softmax: exp(x[i]) / (exp(x[0]) + exp(x[1]) + ...) ----

			// Turns scores into probabilities: all results are in [0, 1] and sum to 1.
			// Computed with exp(x[i] - max), for the same reason as LogSumExp.
			static TVector<T> Softmax(const TVector<T>& vec);
			// Softmax of every row (axisCols: the columns are the classes, each row sums
			// to 1) or of every column (axisRows). The result has the size of 'matrix'.
			static TMatrix<T> Softmax(const TMatrix<T>& matrix, EAxis axis);

		private:
			static_assert(std::is_floating_point<T>::value, "TStatistics: T must be float or double");
		};

		typedef TStatistics<double> CStatistics;
		typedef TStatistics<float> CStatisticsF;
	}
}
