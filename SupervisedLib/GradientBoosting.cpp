#include "GradientBoosting.h"
#include "ModelUtils.h"

#include <numeric>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace supervised
	{
		CGradientBoostingRegressor::CGradientBoostingRegressor(const SGradientBoostingOptions& options) : m_options(options)
		{
			if (options.treeCount == 0)
				throw CommonLib::CExcBase("Gradient boosting: the number of trees must be at least 1");

			if (!(options.learningRate > 0))
				throw CommonLib::CExcBase("Gradient boosting: the learning rate {0} must be positive", options.learningRate);
		}

		void CGradientBoostingRegressor::Fit(const CMatrix& x, const CVector& y)
		{
			detail::CheckFit(x, y.Size(), "Gradient boosting");

			const size_t nRows = x.Rows();

			m_bFitted = false;
			m_dInitial = y.Mean();

			std::vector<size_t> samples(nRows);
			std::iota(samples.begin(), samples.end(), size_t(0));

			// What the model so far gets wrong; starts as y - mean.
			std::vector<double> residuals(nRows);
			for (size_t i = 0; i < nRows; ++i)
				residuals[i] = y[i] - m_dInitial;

			std::mt19937 rng(m_options.tree.seed);

			m_trees.assign(m_options.treeCount, CTree());
			for (CTree& tree : m_trees)
			{
				tree.Build(x, residuals, 0, samples, m_options.tree, rng);

				for (size_t i = 0; i < nRows; ++i)
					residuals[i] -= m_options.learningRate * *tree.Predict(x.Data() + i * x.Cols());
			}

			m_nFeatures = x.Cols();
			m_bFitted = true;
		}

		CVector CGradientBoostingRegressor::Predict(const CMatrix& x) const
		{
			return Predict(x, m_trees.size());
		}

		CVector CGradientBoostingRegressor::Predict(const CMatrix& x, size_t nTrees) const
		{
			detail::CheckPredict(x, m_nFeatures, m_bFitted, "Gradient boosting");

			if (nTrees > m_trees.size())
				throw CommonLib::CExcBase("Gradient boosting: {0} trees asked, the model has {1}", nTrees, m_trees.size());

			CVector result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
			{
				const double* pRow = x.Data() + i * x.Cols();

				double sum = 0;
				for (size_t k = 0; k < nTrees; ++k)
					sum += *m_trees[k].Predict(pRow);

				result[i] = m_dInitial + m_options.learningRate * sum;
			}

			return result;
		}
	}
}
