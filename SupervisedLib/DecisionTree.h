#pragma once

#include <random>
#include <vector>

#include "Model.h"

namespace ML
{
	namespace supervised
	{
		struct SDecisionTreeOptions
		{
			// The longest path from the root to a leaf, in splits. 0: no limit.
			size_t maxDepth = 0;
			// A node with fewer samples becomes a leaf.
			size_t minSamplesSplit = 2;
			// A split must leave at least this many samples on each side.
			size_t minSamplesLeaf = 1;
			// The number of features, picked at random, that are tried at every split.
			// 0: all of them.
			size_t maxFeatures = 0;
			// Seed of the random numbers (the choice of features; in the ensembles also
			// the choice of samples). The same seed and data give the same model.
			unsigned int seed = 0;
		};

		// A CART tree: every inner node tests "feature <= threshold" and sends a row left
		// (true) or right. A leaf holds the answer: the class shares of its training
		// samples (classification) or their mean target (regression).
		//
		// The split of a node is the feature and threshold that make the two sides as
		// pure as possible: the smallest Gini impurity for classes, the smallest squared
		// error for values. Thresholds are halfway between two neighboring values of a
		// feature.
		//
		// This class is the tree itself, shared by the decision tree, random forest and
		// gradient boosting models below; use those.
		class CTree
		{
		public:
			// 'targets' has one value per row of 'x': a class number for classification
			// (nClasses > 0) or the target value for regression (nClasses == 0).
			// 'samples' lists the rows to learn from; a row can be listed several times.
			void Build(const CMatrix& x, const std::vector<double>& targets, size_t nClasses,
				const std::vector<size_t>& samples, const SDecisionTreeOptions& options, std::mt19937& rng);

			// The values of the leaf a row falls in: GetOutputCount() class shares, or
			// one mean. 'pRow' points to the feature values of the row.
			const double* Predict(const double* pRow) const;

			// The number of classes, or 1 for regression.
			size_t GetOutputCount() const { return m_nOutputs; }
			size_t GetNodeCount() const { return m_nodes.size(); }
			size_t GetDepth() const { return m_nDepth; }

		private:
			struct SNode
			{
				bool leaf = true;
				size_t feature = 0;
				double threshold = 0;
				size_t left = 0;	// index of the node for feature <= threshold
				size_t right = 0;	// index of the node for feature > threshold
			};

			std::vector<SNode> m_nodes;
			std::vector<double> m_values;	// GetOutputCount() values per node
			size_t m_nOutputs = 0;
			size_t m_nDepth = 0;
		};

		class CDecisionTreeClassifier : public IClassifier
		{
		public:
			explicit CDecisionTreeClassifier(const SDecisionTreeOptions& options = SDecisionTreeOptions());

			void Fit(const CMatrix& x, const CVectorI& y) override;
			CVectorI Predict(const CMatrix& x) const override;
			// Rows() x classes: the class shares in the leaf of every row.
			CMatrix PredictProbability(const CMatrix& x) const;

			size_t GetDepth() const { return m_tree.GetDepth(); }
			size_t GetNodeCount() const { return m_tree.GetNodeCount(); }

		private:
			SDecisionTreeOptions m_options;
			CTree m_tree;
			size_t m_nFeatures = 0;
			bool m_bFitted = false;
		};

		class CDecisionTreeRegressor : public IRegressor
		{
		public:
			explicit CDecisionTreeRegressor(const SDecisionTreeOptions& options = SDecisionTreeOptions());

			void Fit(const CMatrix& x, const CVector& y) override;
			CVector Predict(const CMatrix& x) const override;

			size_t GetDepth() const { return m_tree.GetDepth(); }
			size_t GetNodeCount() const { return m_tree.GetNodeCount(); }

		private:
			SDecisionTreeOptions m_options;
			CTree m_tree;
			size_t m_nFeatures = 0;
			bool m_bFitted = false;
		};
	}
}
