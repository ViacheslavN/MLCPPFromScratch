#pragma once

#include <cstddef>

#include "NumericsCore/Matrix.h"
#include "NumericsCore/Vector.h"

namespace ML
{
	namespace supervised
	{
		using numerics::CMatrix;
		using numerics::CVector;
		using numerics::CVectorI;

		// The common interface of the models. In every method 'x' has one row per sample
		// and one column per feature. Wrong sizes, a Predict before a Fit and data a model
		// cannot handle throw CommonLib::CExcBase.

		// A model that predicts a number.
		class IRegressor
		{
		public:
			virtual ~IRegressor() = default;

			// Learns from the samples 'x' and their target values 'y' (one per row of 'x').
			// A second Fit starts from scratch.
			virtual void Fit(const CMatrix& x, const CVector& y) = 0;
			// The predicted value for every row of 'x'.
			virtual CVector Predict(const CMatrix& x) const = 0;

			// R^2, the coefficient of determination of Predict(x) against 'y':
			// 1 is a perfect fit, 0 is no better than always predicting the mean of 'y'.
			double Score(const CMatrix& x, const CVector& y) const;
		};

		// A model that predicts a class. The classes are numbered 0, 1, ..., k - 1; the
		// number of classes is the largest label in 'y' plus one, and must be at least 2.
		class IClassifier
		{
		public:
			virtual ~IClassifier() = default;

			// Learns from the samples 'x' and their class labels 'y' (one per row of 'x').
			// A second Fit starts from scratch.
			virtual void Fit(const CMatrix& x, const CVectorI& y) = 0;
			// The predicted class for every row of 'x'.
			virtual CVectorI Predict(const CMatrix& x) const = 0;

			// Accuracy: the share of the rows of 'x' whose predicted class equals 'y'.
			double Score(const CMatrix& x, const CVectorI& y) const;
		};
	}
}
