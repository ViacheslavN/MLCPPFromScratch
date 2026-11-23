#pragma once

#include "../Model.h"

namespace ML
{
	namespace supervised
	{
		struct SLogisticRegressionOptions
		{
			double learningRate = 0.1;
			size_t maxIterations = 10000;
			// Stops when every component of the gradient is smaller than this.
			double tolerance = 1e-6;
			// L2 penalty on the weights (not on the intercepts): adds
			// (l2 / 2) * sum of weights^2 to the loss. 0: no penalty.
			double l2 = 0;
		};

		// Logistic regression for two or more classes (softmax regression): one linear
		// score per class, turned into probabilities by softmax. Trained by batch gradient
		// descent on the mean cross-entropy. Scale the features first (zero mean, unit
		// variance), as for any gradient descent.
		//
		// With two classes this is the classic binary logistic regression: the
		// probability of class 1 is sigmoid(score1 - score0).
		class CLogisticRegression : public IClassifier
		{
		public:
			explicit CLogisticRegression(const SLogisticRegressionOptions& options = SLogisticRegressionOptions());

			void Fit(const CMatrix& x, const CVectorI& y) override;
			using IClassifier::Fit;		// the overload for a vector of one feature
			CVectorI Predict(const CMatrix& x) const override;
			using IClassifier::Predict;
			// Rows() x classes: the probability of every class for every row; a row sums to 1.
			CMatrix PredictProbability(const CMatrix& x) const;

			// classes x features: row k holds the weights of class k.
			const CMatrix& GetWeights() const { return m_weights; }
			// One intercept per class.
			const CVector& GetIntercepts() const { return m_intercepts; }
			size_t GetClassCount() const { return m_weights.Rows(); }

			// The iterations made, and whether the fit stopped on the tolerance (true) or
			// on maxIterations (false). With separable classes and no l2 penalty the
			// weights grow without limit, so the fit ends on maxIterations.
			size_t GetIterations() const { return m_nIterations; }
			bool IsConverged() const { return m_bConverged; }

		private:
			CMatrix Scores(const CMatrix& x) const;

		private:
			SLogisticRegressionOptions m_options;
			CMatrix m_weights;
			CVector m_intercepts;
			size_t m_nIterations = 0;
			bool m_bConverged = false;
			bool m_bFitted = false;
		};
	}
}
