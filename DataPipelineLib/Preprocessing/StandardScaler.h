#pragma once

#include "NumericsCoreLib/Matrix.h"
#include "NumericsCoreLib/Vector.h"

namespace ML
{
	namespace data
	{
		// Standard scaling of features: every column becomes
		//   (value - mean of the column) / standard deviation of the column,
		// so it has mean 0 and standard deviation 1. Models that compare features by
		// size need it: k-NN and SVM (distances), gradient descent (one learning rate
		// for all weights), ridge and lasso (one penalty for all weights). Trees do not.
		//
		// Fit learns the means and deviations, Transform applies them. Fit on the
		// training data only and transform the test data with the same scaler:
		// statistics taken from the test data would leak it into the model.
		//
		// T is float or double (the member functions are compiled for these two in
		// StandardScaler.cpp). CStandardScaler is the double version, CStandardScalerF
		// the float one. Wrong sizes and a Transform before a Fit throw
		// CommonLib::CExcBase.
		template<class T>
		class TStandardScaler
		{
		public:
			// Learns the mean and the standard deviation of every column of 'x' (one
			// row per sample). Needs at least one row. A second Fit starts from scratch.
			void Fit(const numerics::TMatrix<T>& x);

			// 'x' scaled with the learned values; it must have the columns of the Fit.
			// The result uses the allocator of 'x'.
			numerics::TMatrix<T> Transform(const numerics::TMatrix<T>& x) const;
			// Fit(x), then Transform(x): for the training data.
			numerics::TMatrix<T> FitTransform(const numerics::TMatrix<T>& x);
			// Back to the original values: value * deviation + mean.
			numerics::TMatrix<T> InverseTransform(const numerics::TMatrix<T>& x) const;

			// The learned mean of every column.
			const numerics::TVector<T>& GetMeans() const { return m_means; }
			// What every column is divided by: its standard deviation (of the samples
			// as a whole population, divided by n), or 1 for a column where all values
			// are equal, which then becomes all zeros.
			const numerics::TVector<T>& GetScales() const { return m_scales; }
			bool IsFitted() const { return m_bFitted; }

		private:
			void Check(const numerics::TMatrix<T>& x) const;

		private:
			numerics::TVector<T> m_means;
			numerics::TVector<T> m_scales;
			bool m_bFitted = false;
		};

		typedef TStandardScaler<double> CStandardScaler;
		typedef TStandardScaler<float> CStandardScalerF;
	}
}
