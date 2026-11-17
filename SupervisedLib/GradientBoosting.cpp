#include "GradientBoosting.h"
#include "ModelUtils.h"

#include <numeric>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace supervised
	{
		template<class T>
		TGradientBoostingRegressor<T>::TGradientBoostingRegressor(const SGradientBoostingOptions& options) : m_options(options)
		{
			if (options.treeCount == 0)
				throw CommonLib::CExcBase("Gradient boosting: the number of trees must be at least 1");

			if (!(options.learningRate > 0))
				throw CommonLib::CExcBase("Gradient boosting: the learning rate {0} must be positive", options.learningRate);
		}

		template<class T>
		void TGradientBoostingRegressor<T>::Fit(const TMatrix<T>& x, const TVector<T>& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Gradient boosting");

			const size_t nRows = x.Rows();

			m_bFitted = false;
			m_initial = y.Mean();

			std::vector<size_t> samples(nRows);
			std::iota(samples.begin(), samples.end(), size_t(0));

			// What the model so far gets wrong; starts as y - mean.
			std::vector<T> residuals(nRows);
			for (size_t i = 0; i < nRows; ++i)
				residuals[i] = y[i] - m_initial;

			std::mt19937 rng(m_options.tree.seed);
			const T learningRate = T(m_options.learningRate);

			m_trees.assign(m_options.treeCount, TTree<T>());
			for (TTree<T>& tree : m_trees)
			{
				tree.Build(x, residuals, 0, samples, m_options.tree, rng);

				for (size_t i = 0; i < nRows; ++i)
					residuals[i] -= learningRate * *tree.Predict(x.Data() + i * x.Cols());
			}

			m_nFeatures = x.Cols();
			m_bFitted = true;
		}

		template<class T>
		TVector<T> TGradientBoostingRegressor<T>::Predict(const TMatrix<T>& x) const
		{
			return Predict(x, m_trees.size());
		}

		template<class T>
		TVector<T> TGradientBoostingRegressor<T>::Predict(const TMatrix<T>& x, size_t nTrees) const
		{
			detail::CheckPredict(x.Cols(), m_nFeatures, m_bFitted, "Gradient boosting");

			if (nTrees > m_trees.size())
				throw CommonLib::CExcBase("Gradient boosting: {0} trees asked, the model has {1}", nTrees, m_trees.size());

			TVector<T> result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
			{
				const T* pRow = x.Data() + i * x.Cols();

				T sum = 0;
				for (size_t k = 0; k < nTrees; ++k)
					sum += *m_trees[k].Predict(pRow);

				result[i] = m_initial + T(m_options.learningRate) * sum;
			}

			return result;
		}

		// The element types the library supports.
		template class TGradientBoostingRegressor<float>;
		template class TGradientBoostingRegressor<double>;
	}
}