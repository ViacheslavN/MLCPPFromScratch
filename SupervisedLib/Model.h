#pragma once

#include <cstddef>

#include "NumericsCoreLib/Matrix.h"
#include "NumericsCoreLib/Vector.h"

namespace ML
{
	namespace supervised
	{
		using numerics::TMatrix;
		using numerics::TVector;
		using numerics::CMatrix;
		using numerics::CVector;
		using numerics::CVectorI;

		// The common interface of the models. In every method 'x' has one row per sample
		// and one column per feature. Wrong sizes, a Predict before a Fit and data a model
		// cannot handle throw CommonLib::CExcBase.
		//
		// When the data has a single feature, 'x' can be a vector instead: one element
		// per sample. It is the same as a matrix with one column, so a model fitted with
		// a vector predicts from a vector or from a one-column matrix alike.

		// A model that predicts a number. T is the type of the data and of the model's own
		// numbers: float or double (the member functions of the models are compiled for
		// these two).
		template<class T>
		class IRegressorT
		{
		public:
			typedef T TValue;

			virtual ~IRegressorT() = default;

			// Learns from the samples 'x' and their target values 'y' (one per row of 'x').
			// A second Fit starts from scratch.
			virtual void Fit(const TMatrix<T>& x, const TVector<T>& y) = 0;
			// The predicted value for every row of 'x'.
			virtual TVector<T> Predict(const TMatrix<T>& x) const = 0;

			// R^2, the coefficient of determination of Predict(x) against 'y':
			// 1 is a perfect fit, 0 is no better than always predicting the mean of 'y'.
			double Score(const TMatrix<T>& x, const TVector<T>& y) const;

			// The same for data with one feature: 'x' holds its value for every sample.
			void Fit(const TVector<T>& x, const TVector<T>& y);
			TVector<T> Predict(const TVector<T>& x) const;
			double Score(const TVector<T>& x, const TVector<T>& y) const;
		};

		typedef IRegressorT<double> IRegressor;
		typedef IRegressorT<float> IRegressorF;

		// A model that predicts a class. The classes are numbered 0, 1, ..., k - 1; the
		// number of classes is the largest label in 'y' plus one, and must be at least 2.
		// The classifiers work with double only.
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

			// The same for data with one feature: 'x' holds its value for every sample.
			void Fit(const CVector& x, const CVectorI& y);
			CVectorI Predict(const CVector& x) const;
			double Score(const CVector& x, const CVectorI& y) const;
		};
	}
}
