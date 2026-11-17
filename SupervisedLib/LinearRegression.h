#pragma once

#include "Model.h"

namespace ML
{
	namespace supervised
	{
		// The linear models are templates: T is float or double (the member functions are
		// compiled for these two in LinearRegression.cpp). The C... names at the end of
		// the file are the double versions, the C...F names the float ones.

		// The base of the linear models: prediction = x . weights + intercept.
		// The classes below differ in how Fit finds the weights.
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

		typedef TLinearRegression<double> CLinearRegression;
		typedef TRidgeRegression<double> CRidgeRegression;
		typedef TLassoRegression<double> CLassoRegression;
		typedef TLinearRegression<float> CLinearRegressionF;
		typedef TRidgeRegression<float> CRidgeRegressionF;
		typedef TLassoRegression<float> CLassoRegressionF;
	}
}