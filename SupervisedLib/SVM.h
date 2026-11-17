#pragma once

#include "Model.h"

namespace ML
{
	namespace supervised
	{
		struct SSVMOptions
		{
			enum EKernel
			{
				kernelLinear,	// K(a, b) = a . b: a straight boundary
				kernelRbf		// K(a, b) = exp(-gamma * |a - b|^2): a curved boundary
			};

			EKernel kernel = kernelRbf;
			// The price of a sample on the wrong side of its margin. Large: fit the
			// training data closely; small: a wider margin and a simpler boundary.
			double c = 1.0;
			// RBF kernel only: how fast the influence of a sample fades with distance.
			// 0: 1 / (number of features * variance of all values of x).
			double gamma = 0;
			// Stops when no pair of samples violates the optimality conditions by more than this.
			double tolerance = 1e-3;
			size_t maxIterations = 100000;
		};

		// Support vector machine for two classes (labels 0 and 1), trained by SMO
		// (sequential minimal optimization).
		//
		// The SVM looks for the boundary with the widest margin between the classes. In
		// its dual form the unknowns are one multiplier (alpha) per training sample,
		// tied by one equality constraint, so the smallest step that keeps the
		// constraint changes two of them. SMO repeats exactly that: it picks the pair
		// that violates the optimality conditions the most and solves for the two
		// alphas in closed form. The samples that end with alpha > 0 are the support
		// vectors; only they are kept for the predictions.
		//
		// Scale the features first. Fit computes the kernel between all pairs of
		// samples: n * n values for n samples.
		class CSVMClassifier : public IClassifier
		{
		public:
			explicit CSVMClassifier(const SSVMOptions& options = SSVMOptions());

			// Throws when 'y' has labels other than 0 and 1.
			void Fit(const CMatrix& x, const CVectorI& y) override;
			CVectorI Predict(const CMatrix& x) const override;
			// The signed score of every row: positive for class 1, negative for class 0;
			// about +1 and -1 on the margins.
			CVector DecisionFunction(const CMatrix& x) const;

			// The support vectors (one per row) and, for each, alpha times its class
			// sign (+1 for class 1, -1 for class 0).
			const CMatrix& GetSupportVectors() const { return m_supportVectors; }
			const CVector& GetCoefficients() const { return m_coefficients; }
			double GetIntercept() const { return m_dIntercept; }

			// The iterations made, and whether the fit stopped on the tolerance (true)
			// or on maxIterations (false).
			size_t GetIterations() const { return m_nIterations; }
			bool IsConverged() const { return m_bConverged; }

		private:
			double Kernel(const double* pA, const double* pB, size_t nSize) const;

		private:
			SSVMOptions m_options;
			double m_dGamma = 0;
			CMatrix m_supportVectors;
			CVector m_coefficients;
			double m_dIntercept = 0;
			size_t m_nFeatures = 0;
			size_t m_nIterations = 0;
			bool m_bConverged = false;
			bool m_bFitted = false;
		};
	}
}
