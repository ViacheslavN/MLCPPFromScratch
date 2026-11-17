#include "KNearestNeighbors.h"
#include "ModelUtils.h"

#include <utility>
#include <vector>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace supervised
	{
		namespace
		{
			// (squared distance, index of the training sample)
			typedef std::vector<std::pair<double, size_t>> TNeighbors;

			void CheckNeighbors(size_t nNeighbors)
			{
				if (nNeighbors == 0)
					throw CommonLib::CExcBase("k-NN: the number of neighbors must be at least 1");
			}

			void CheckSamples(size_t nSamples, size_t nNeighbors)
			{
				if (nSamples < nNeighbors)
					throw CommonLib::CExcBase("k-NN: {0} samples are not enough for {1} neighbors", nSamples, nNeighbors);
			}

			// Puts the nNeighbors training samples closest to 'pRow' in front of
			// 'neighbors', the closest first. The squared distance orders the samples
			// like the distance, without the square root.
			void FindNeighbors(const CMatrix& train, const double* pRow, size_t nNeighbors, TNeighbors& neighbors)
			{
				const size_t nRows = train.Rows();
				const size_t nCols = train.Cols();

				neighbors.resize(nRows);
				for (size_t i = 0; i < nRows; ++i)
				{
					const double* pTrain = train.Data() + i * nCols;

					double distance = 0;
					for (size_t j = 0; j < nCols; ++j)
						distance += (pRow[j] - pTrain[j]) * (pRow[j] - pTrain[j]);

					neighbors[i] = std::make_pair(distance, i);
				}

				// Pairs compare by distance, then by index: equal distances are ordered.
				std::partial_sort(neighbors.begin(), neighbors.begin() + nNeighbors, neighbors.end());
			}
		}

		// ---- CKNNClassifier ----

		CKNNClassifier::CKNNClassifier(size_t nNeighbors) : m_nNeighbors(nNeighbors)
		{
			CheckNeighbors(nNeighbors);
		}

		void CKNNClassifier::Fit(const CMatrix& x, const CVectorI& y)
		{
			detail::CheckFit(x, y.Size(), "k-NN");
			CheckSamples(x.Rows(), m_nNeighbors);

			m_bFitted = false;
			m_nClasses = detail::CountClasses(y, "k-NN");
			m_x = x;
			m_y = y;
			m_bFitted = true;
		}

		CVectorI CKNNClassifier::Predict(const CMatrix& x) const
		{
			const CMatrix probabilities = PredictProbability(x);

			CVectorI result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
				result[i] = int(detail::ArgMax(probabilities.Data() + i * m_nClasses, m_nClasses));

			return result;
		}

		CMatrix CKNNClassifier::PredictProbability(const CMatrix& x) const
		{
			detail::CheckPredict(x, m_x.Cols(), m_bFitted, "k-NN");

			CMatrix result(x.Rows(), m_nClasses, x.GetAlloc());
			TNeighbors neighbors;
			const double vote = 1.0 / double(m_nNeighbors);

			for (size_t i = 0; i < x.Rows(); ++i)
			{
				FindNeighbors(m_x, x.Data() + i * x.Cols(), m_nNeighbors, neighbors);

				for (size_t k = 0; k < m_nNeighbors; ++k)
					result(i, size_t(m_y[neighbors[k].second])) += vote;
			}

			return result;
		}

		// ---- CKNNRegressor ----

		CKNNRegressor::CKNNRegressor(size_t nNeighbors) : m_nNeighbors(nNeighbors)
		{
			CheckNeighbors(nNeighbors);
		}

		void CKNNRegressor::Fit(const CMatrix& x, const CVector& y)
		{
			detail::CheckFit(x, y.Size(), "k-NN");
			CheckSamples(x.Rows(), m_nNeighbors);

			m_bFitted = false;
			m_x = x;
			m_y = y;
			m_bFitted = true;
		}

		CVector CKNNRegressor::Predict(const CMatrix& x) const
		{
			detail::CheckPredict(x, m_x.Cols(), m_bFitted, "k-NN");

			CVector result(x.Rows(), x.GetAlloc());
			TNeighbors neighbors;

			for (size_t i = 0; i < x.Rows(); ++i)
			{
				FindNeighbors(m_x, x.Data() + i * x.Cols(), m_nNeighbors, neighbors);

				double sum = 0;
				for (size_t k = 0; k < m_nNeighbors; ++k)
					sum += m_y[neighbors[k].second];

				result[i] = sum / double(m_nNeighbors);
			}

			return result;
		}
	}
}
