#pragma once

#include <vector>

#include "DecisionTree.h"

namespace ML
{
	namespace supervised
	{
		struct SGradientBoostingOptions
		{
			SGradientBoostingOptions() { tree.maxDepth = 3; }

			size_t treeCount = 100;
			// The share of every tree's prediction that is added to the model. Smaller
			// values need more trees and usually generalize better.
			double learningRate = 0.1;
			// The options of the trees: shallow by default (depth 3).
			SDecisionTreeOptions tree;
		};

		// Gradient boosting for regression with the squared error. The model starts as
		// the mean of the targets. Every next tree is fitted to the residuals, what the
		// model so far gets wrong (for the squared error the residual is the negative
		// gradient of the loss, hence the name), and a share of it is added to the model:
		//   prediction = mean + learningRate * (tree1 + tree2 + ...).
		// Unlike a forest, the trees are small and each one depends on the ones before.
		class CGradientBoostingRegressor : public IRegressor
		{
		public:
			explicit CGradientBoostingRegressor(const SGradientBoostingOptions& options = SGradientBoostingOptions());

			void Fit(const CMatrix& x, const CVector& y) override;
			CVector Predict(const CMatrix& x) const override;
			// The prediction of the first 'nTrees' trees only: shows how the error
			// changes as trees are added.
			CVector Predict(const CMatrix& x, size_t nTrees) const;

			size_t GetTreeCount() const { return m_trees.size(); }

		private:
			SGradientBoostingOptions m_options;
			std::vector<CTree> m_trees;
			double m_dInitial = 0;
			size_t m_nFeatures = 0;
			bool m_bFitted = false;
		};
	}
}
