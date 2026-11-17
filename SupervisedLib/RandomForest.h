#pragma once

#include <vector>

#include "DecisionTree.h"

namespace ML
{
	namespace supervised
	{
		struct SRandomForestOptions
		{
			size_t treeCount = 100;
			// True: every tree learns from a bootstrap sample, as many rows as the data
			// has, drawn at random with repetition. False: every tree sees all rows.
			bool bootstrap = true;
			// The options of the trees; 'tree.seed' seeds the whole forest. Here
			// tree.maxFeatures == 0 means the usual default of a forest: the square root
			// of the number of features for classification, a third for regression.
			SDecisionTreeOptions tree;
		};

		// A random forest: many deep trees, each grown on a random sample of the rows
		// and choosing every split among a random subset of the features. One tree
		// overfits; the trees overfit differently, and averaging them cancels most of it.

		// Predicts the class with the largest mean share over the trees.
		class CRandomForestClassifier : public IClassifier
		{
		public:
			explicit CRandomForestClassifier(const SRandomForestOptions& options = SRandomForestOptions());

			void Fit(const CMatrix& x, const CVectorI& y) override;
			using IClassifier::Fit;		// the overload for a vector of one feature
			CVectorI Predict(const CMatrix& x) const override;
			using IClassifier::Predict;
			// Rows() x classes: the class shares averaged over the trees.
			CMatrix PredictProbability(const CMatrix& x) const;

			size_t GetTreeCount() const { return m_trees.size(); }

		private:
			SRandomForestOptions m_options;
			std::vector<CTree> m_trees;
			size_t m_nFeatures = 0;
			size_t m_nClasses = 0;
			bool m_bFitted = false;
		};

		// Predicts the mean of the predictions of the trees.
		// T is float or double; CRandomForestRegressor is the double version,
		// CRandomForestRegressorF the float one.
		template<class T>
		class TRandomForestRegressor : public IRegressorT<T>
		{
		public:
			explicit TRandomForestRegressor(const SRandomForestOptions& options = SRandomForestOptions());

			void Fit(const TMatrix<T>& x, const TVector<T>& y) override;
			using IRegressorT<T>::Fit;		// the overload for a vector of one feature
			TVector<T> Predict(const TMatrix<T>& x) const override;
			using IRegressorT<T>::Predict;

			size_t GetTreeCount() const { return m_trees.size(); }

		private:
			SRandomForestOptions m_options;
			std::vector<TTree<T>> m_trees;
			size_t m_nFeatures = 0;
			bool m_bFitted = false;
		};

		typedef TRandomForestRegressor<double> CRandomForestRegressor;
		typedef TRandomForestRegressor<float> CRandomForestRegressorF;
	}
}