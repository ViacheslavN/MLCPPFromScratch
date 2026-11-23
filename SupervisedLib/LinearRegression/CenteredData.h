#pragma once

// Internal helper of the linear models; not part of the interface of SupervisedLib.

#include "../Model.h"

namespace ML
{
	namespace supervised
	{
		namespace detail
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
	}
}
