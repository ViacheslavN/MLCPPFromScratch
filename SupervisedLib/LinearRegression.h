#pragma once

#include "Model.h"

namespace ML
{
	namespace supervised
	{
		// The base of the linear models: prediction = x . weights + intercept.
		// The classes below differ in how Fit finds the weights.
		class CLinearModel : public IRegressor
		{
		public:
			CVector Predict(const CMatrix& x) const override;

			// One weight per feature.
			const CVector& GetWeights() const { return m_weights; }
			double GetIntercept() const { return m_dIntercept; }

		protected:
			CVector m_weights;
			double m_dIntercept = 0;
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
			// Stops when every component of the gradient is smaller than this.
			double tolerance = 1e-8;
		};

		// Ordinary least squares: minimizes the sum of (prediction - y)^2.
		class CLinearRegression : public CLinearModel
		{
		public:
			explicit CLinearRegression(const SLinearRegressionOptions& options = SLinearRegressionOptions());

			// The normal equation throws when the features are linearly dependent (use
			// CRidgeRegression then). Gradient descent throws when it diverges.
			void Fit(const CMatrix& x, const CVector& y) override;

			// Gradient descent: the iterations made, and whether it stopped on the
			// tolerance (true) or on maxIterations (false).
			size_t GetIterations() const { return m_nIterations; }
			bool IsConverged() const { return m_bConverged; }

		private:
			void FitNormalEquation(const CMatrix& x, const CVector& y);
			void FitGradientDescent(const CMatrix& x, const CVector& y);

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
		class CRidgeRegression : public CLinearModel
		{
		public:
			// 'alpha' >= 0; 0 is ordinary least squares.
			explicit CRidgeRegression(double alpha = 1.0);

			void Fit(const CMatrix& x, const CVector& y) override;

		private:
			double m_dAlpha;
		};

		// Least squares with an L1 penalty: minimizes
		//   sum of (prediction - y)^2 / (2 * n) + alpha * sum of |weights|
		// for n samples. The penalty sets the weights of unhelpful features to exactly
		// zero. The intercept is not penalized. Solved by coordinate descent.
		class CLassoRegression : public CLinearModel
		{
		public:
			// 'alpha' >= 0. Stops when no weight changes by more than 'tolerance' in a
			// pass over the features, or after 'maxIterations' passes.
			explicit CLassoRegression(double alpha = 1.0, size_t maxIterations = 1000, double tolerance = 1e-8);

			void Fit(const CMatrix& x, const CVector& y) override;

			size_t GetIterations() const { return m_nIterations; }
			bool IsConverged() const { return m_bConverged; }

		private:
			double m_dAlpha;
			size_t m_nMaxIterations;
			double m_dTolerance;
			size_t m_nIterations = 0;
			bool m_bConverged = false;
		};
	}
}
