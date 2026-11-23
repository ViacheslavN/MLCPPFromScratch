#pragma once

#include "../Model.h"

namespace ML
{
	namespace supervised
	{
		// The linear models are templates: T is float or double (the member functions are
		// compiled for these two in the .cpp files). The C... names at the end of each
		// header are the double versions, the C...F names the float ones.

		// The base of the linear models: prediction = x . weights + intercept.
		// TLinearRegression (below), TRidgeRegression (RidgeRegression.h) and
		// TLassoRegression (LassoRegression.h) differ in how Fit finds the weights.
		template<class T>
		class TLinearModel : public IRegressorT<T>
		{
		public:
			TVector<T> Predict(const TMatrix<T>& x) const override;
			using IRegressorT<T>::Predict;

			// One weight per feature.
			const TVector<T>& GetWeights() const { return m_weights; }
			T GetIntercept() const { return m_intercept; }

		protected:
			TVector<T> m_weights;
			T m_intercept = 0;
			bool m_bFitted = false;
		};

		struct SLinearRegressionOptions
		{
			enum ESolver
			{
				// Exact least squares solution in one step (QR decomposition).
				solverNormalEquation,
				// Batch gradient descent on the mean squared error. Scale the features
				// first (zero mean, unit variance): it converges slowly or not at all
				// when they have very different ranges.
				solverGradientDescent
			};

			ESolver solver = solverNormalEquation;

			// Gradient descent only.
			double learningRate = 0.01;
			size_t maxIterations = 10000;
			// Stops when every component of the gradient is smaller than this. For float
			// use about 1e-4: its rounding keeps the gradient above 1e-8.
			double tolerance = 1e-8;
		};

		// Ordinary least squares: minimizes the sum of (prediction - y)^2.
		template<class T>
		class TLinearRegression : public TLinearModel<T>
		{
		public:
			explicit TLinearRegression(const SLinearRegressionOptions& options = SLinearRegressionOptions());

			// The normal equation throws when the features are linearly dependent (use
			// TRidgeRegression then). Gradient descent throws when it diverges.
			void Fit(const TMatrix<T>& x, const TVector<T>& y) override;
			using IRegressorT<T>::Fit;		// the overload for a vector of one feature

			// Gradient descent: the iterations made, and whether it stopped on the
			// tolerance (true) or on maxIterations (false).
			size_t GetIterations() const { return m_nIterations; }
			bool IsConverged() const { return m_bConverged; }

		private:
			void FitNormalEquation(const TMatrix<T>& x, const TVector<T>& y);
			void FitGradientDescent(const TMatrix<T>& x, const TVector<T>& y);

		private:
			SLinearRegressionOptions m_options;
			size_t m_nIterations = 0;
			bool m_bConverged = false;
		};

		typedef TLinearRegression<double> CLinearRegression;
		typedef TLinearRegression<float> CLinearRegressionF;
	}
}
