#pragma once

#include "Model.h"

namespace ML
{
	namespace supervised
	{
		// k nearest neighbors: nothing is learned in Fit, the training samples are kept.
		// A prediction looks at the k training samples closest to the row (Euclidean
		// distance; on equal distances the earlier training sample wins). Scale the
		// features first: the distance treats all of them alike, so a feature with a
		// large range dominates.
		//
		// Every prediction compares the row with all training samples: n * m distances
		// for m rows and n training samples.

		// The class most common among the k neighbors (the smaller label on a tie).
		class CKNNClassifier : public IClassifier
		{
		public:
			explicit CKNNClassifier(size_t nNeighbors = 5);

			// Throws when there are fewer samples than neighbors.
			void Fit(const CMatrix& x, const CVectorI& y) override;
			CVectorI Predict(const CMatrix& x) const override;
			// Rows() x classes: the share of the k neighbors that have each class.
			CMatrix PredictProbability(const CMatrix& x) const;

		private:
			size_t m_nNeighbors;
			size_t m_nClasses = 0;
			CMatrix m_x;
			CVectorI m_y;
			bool m_bFitted = false;
		};

		// The mean target of the k neighbors.
		class CKNNRegressor : public IRegressor
		{
		public:
			explicit CKNNRegressor(size_t nNeighbors = 5);

			// Throws when there are fewer samples than neighbors.
			void Fit(const CMatrix& x, const CVector& y) override;
			CVector Predict(const CMatrix& x) const override;

		private:
			size_t m_nNeighbors;
			CMatrix m_x;
			CVector m_y;
			bool m_bFitted = false;
		};
	}
}
