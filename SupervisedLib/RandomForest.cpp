#include "RandomForest.h"
#include "ModelUtils.h"

#include <cmath>
#include <numeric>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace supervised
	{
		namespace
		{
			void CheckOptions(const SRandomForestOptions& options)
			{
				if (options.treeCount == 0)
					throw CommonLib::CExcBase("Random forest: the number of trees must be at least 1");
			}

			// Grows the trees of a forest. 'nDefaultFeatures' replaces tree.maxFeatures == 0.
			template<class T>
			void BuildTrees(std::vector<TTree<T>>& trees, const TMatrix<T>& x, const std::vector<T>& targets, size_t nClasses,
				const SRandomForestOptions& options, size_t nDefaultFeatures)
			{
				SDecisionTreeOptions treeOptions = options.tree;
				if (treeOptions.maxFeatures == 0)
					treeOptions.maxFeatures = (std::max)(nDefaultFeatures, size_t(1));

				const size_t nRows = x.Rows();
				std::vector<size_t> samples(nRows);
				std::iota(samples.begin(), samples.end(), size_t(0));

				// One generator for the whole forest: the trees differ, the forest is
				// the same for the same seed. rng() % n instead of a std distribution,
				// whose results differ between standard libraries.
				std::mt19937 rng(options.tree.seed);

				trees.assign(options.treeCount, TTree<T>());
				for (TTree<T>& tree : trees)
				{
					if (options.bootstrap)
					{
						for (size_t i = 0; i < nRows; ++i)
							samples[i] = rng() % nRows;
					}

					tree.Build(x, targets, nClasses, samples, treeOptions, rng);
				}
			}
		}

		// ---- CRandomForestClassifier ----

		CRandomForestClassifier::CRandomForestClassifier(const SRandomForestOptions& options) : m_options(options)
		{
			CheckOptions(options);
		}

		void CRandomForestClassifier::Fit(const CMatrix& x, const CVectorI& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Random forest");

			const size_t nClasses = detail::CountClasses(y, "Random forest");

			std::vector<double> targets(y.Size());
			for (size_t i = 0; i < y.Size(); ++i)
				targets[i] = double(y[i]);

			m_bFitted = false;

			BuildTrees(m_trees, x, targets, nClasses, m_options, size_t(std::sqrt(double(x.Cols()))));

			m_nFeatures = x.Cols();
			m_nClasses = nClasses;
			m_bFitted = true;
		}

		CVectorI CRandomForestClassifier::Predict(const CMatrix& x) const
		{
			const CMatrix probabilities = PredictProbability(x);

			CVectorI result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
				result[i] = int(detail::ArgMax(probabilities.Data() + i * m_nClasses, m_nClasses));

			return result;
		}

		CMatrix CRandomForestClassifier::PredictProbability(const CMatrix& x) const
		{
			detail::CheckPredict(x.Cols(), m_nFeatures, m_bFitted, "Random forest");

			CMatrix result(x.Rows(), m_nClasses, x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
			{
				const double* pRow = x.Data() + i * x.Cols();
				for (const CTree& tree : m_trees)
				{
					const double* pShares = tree.Predict(pRow);
					for (size_t k = 0; k < m_nClasses; ++k)
						result(i, k) += pShares[k];
				}
			}

			result *= 1.0 / double(m_trees.size());
			return result;
		}

		// ---- TRandomForestRegressor ----

		template<class T>
		TRandomForestRegressor<T>::TRandomForestRegressor(const SRandomForestOptions& options) : m_options(options)
		{
			CheckOptions(options);
		}

		template<class T>
		void TRandomForestRegressor<T>::Fit(const TMatrix<T>& x, const TVector<T>& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Random forest");

			const std::vector<T> targets(y.Data(), y.Data() + y.Size());

			m_bFitted = false;

			BuildTrees(m_trees, x, targets, 0, m_options, x.Cols() / 3);

			m_nFeatures = x.Cols();
			m_bFitted = true;
		}

		template<class T>
		TVector<T> TRandomForestRegressor<T>::Predict(const TMatrix<T>& x) const
		{
			detail::CheckPredict(x.Cols(), m_nFeatures, m_bFitted, "Random forest");

			TVector<T> result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
			{
				const T* pRow = x.Data() + i * x.Cols();

				T sum = 0;
				for (const TTree<T>& tree : m_trees)
					sum += *tree.Predict(pRow);

				result[i] = sum / T(m_trees.size());
			}

			return result;
		}

		// The element types the library supports.
		template class TRandomForestRegressor<float>;
		template class TRandomForestRegressor<double>;
	}
}