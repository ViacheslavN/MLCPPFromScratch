#include "DecisionTree.h"
#include "ModelUtils.h"

#include <limits>
#include <numeric>
#include <utility>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace supervised
	{
		namespace
		{
			// A node waiting to be split: its samples are samples[begin..end).
			struct SWork
			{
				size_t node;
				size_t begin;
				size_t end;
				size_t depth;
			};

			template<class T>
			T SumOfSquares(const std::vector<T>& values)
			{
				T sum = 0;
				for (T value : values)
					sum += value * value;

				return sum;
			}
		}

		// ---- TTree ----

		template<class T>
		void TTree<T>::Build(const TMatrix<T>& x, const std::vector<T>& targets, size_t nClasses,
			const std::vector<size_t>& samples, const SDecisionTreeOptions& options, std::mt19937& rng)
		{
			const size_t nFeatures = x.Cols();
			const bool bClasses = nClasses > 0;
			const size_t nMinLeaf = (std::max)(options.minSamplesLeaf, size_t(1));
			const size_t nTryFeatures = options.maxFeatures == 0 ? nFeatures : (std::min)(options.maxFeatures, nFeatures);

			m_nOutputs = bClasses ? nClasses : 1;
			m_nDepth = 0;
			m_nodes.assign(1, SNode());
			m_values.assign(m_nOutputs, T(0));

			// The statistics of a set of samples: the number of samples in every class,
			// or the sum of the targets. For both, the quality of a split into a left
			// and a right side is
			//   SumOfSquares(left) / nLeft + SumOfSquares(right) / nRight,
			// the larger the better: maximizing it minimizes the weighted Gini impurity
			// (classes) or the squared error around the two means (values).
			const auto addSample = [&](std::vector<T>& statistics, size_t nSample)
			{
				if (bClasses)
					statistics[size_t(targets[nSample])] += 1;
				else
					statistics[0] += targets[nSample];
			};

			std::vector<size_t> order(samples);		// the samples, regrouped node by node
			std::vector<size_t> features(nFeatures);
			std::iota(features.begin(), features.end(), size_t(0));

			std::vector<std::pair<T, size_t>> sorted;	// (feature value, sample)
			std::vector<T> total(m_nOutputs), left(m_nOutputs), right(m_nOutputs);

			// A stack of nodes instead of recursion: a deep tree cannot overflow the call stack.
			std::vector<SWork> stack;
			stack.push_back({0, 0, order.size(), 0});

			while (!stack.empty())
			{
				const SWork work = stack.back();
				stack.pop_back();

				const size_t nCount = work.end - work.begin;
				m_nDepth = (std::max)(m_nDepth, work.depth);

				// The value of the node, and whether its samples all have the same target.
				std::fill(total.begin(), total.end(), T(0));
				bool bPure = true;
				for (size_t i = work.begin; i < work.end; ++i)
				{
					addSample(total, order[i]);
					if (targets[order[i]] != targets[order[work.begin]])
						bPure = false;
				}

				for (size_t k = 0; k < m_nOutputs; ++k)
					m_values[work.node * m_nOutputs + k] = nCount == 0 ? T(0) : total[k] / T(nCount);

				if (bPure || nCount < options.minSamplesSplit || nCount < 2 * nMinLeaf
					|| (options.maxDepth != 0 && work.depth >= options.maxDepth))
					continue;	// stays a leaf

				// The features to try: all, or the first nTryFeatures of a random shuffle.
				if (nTryFeatures < nFeatures)
				{
					for (size_t i = 0; i < nTryFeatures; ++i)
						std::swap(features[i], features[i + rng() % (nFeatures - i)]);
				}

				T bestScore = -std::numeric_limits<T>::infinity();
				size_t nBestFeature = 0;
				T bestThreshold = 0;
				bool bFound = false;

				sorted.resize(nCount);
				for (size_t f = 0; f < nTryFeatures; ++f)
				{
					const size_t nFeature = features[f];

					for (size_t i = 0; i < nCount; ++i)
						sorted[i] = std::make_pair(x(order[work.begin + i], nFeature), order[work.begin + i]);

					std::sort(sorted.begin(), sorted.end());

					// Move the samples to the left side one by one, in the order of the
					// feature; every place where the value changes is a possible threshold.
					std::fill(left.begin(), left.end(), T(0));
					for (size_t i = 0; i + 1 < nCount; ++i)
					{
						addSample(left, sorted[i].second);

						if (sorted[i].first == sorted[i + 1].first)
							continue;

						const size_t nLeft = i + 1;
						const size_t nRight = nCount - nLeft;
						if (nLeft < nMinLeaf || nRight < nMinLeaf)
							continue;

						for (size_t k = 0; k < m_nOutputs; ++k)
							right[k] = total[k] - left[k];

						const T score = SumOfSquares(left) / T(nLeft) + SumOfSquares(right) / T(nRight);
						if (score > bestScore)
						{
							bestScore = score;
							nBestFeature = nFeature;
							bFound = true;

							// Halfway between the two values; when they are so close that
							// the middle rounds up to the larger one, the smaller one.
							bestThreshold = sorted[i].first + (sorted[i + 1].first - sorted[i].first) / 2;
							if (!(bestThreshold < sorted[i + 1].first))
								bestThreshold = sorted[i].first;
						}
					}
				}

				if (!bFound)
					continue;	// every tried feature is constant in this node

				const auto middle = std::partition(order.begin() + work.begin, order.begin() + work.end,
					[&](size_t nSample) { return x(nSample, nBestFeature) <= bestThreshold; });
				const size_t nMiddle = size_t(middle - order.begin());

				const size_t nLeftNode = m_nodes.size();
				m_nodes.resize(nLeftNode + 2);
				m_values.resize(m_nodes.size() * m_nOutputs, T(0));

				SNode& node = m_nodes[work.node];
				node.leaf = false;
				node.feature = nBestFeature;
				node.threshold = bestThreshold;
				node.left = nLeftNode;
				node.right = nLeftNode + 1;

				stack.push_back({nLeftNode, work.begin, nMiddle, work.depth + 1});
				stack.push_back({nLeftNode + 1, nMiddle, work.end, work.depth + 1});
			}
		}

		template<class T>
		const T* TTree<T>::Predict(const T* pRow) const
		{
			size_t nNode = 0;
			while (!m_nodes[nNode].leaf)
				nNode = pRow[m_nodes[nNode].feature] <= m_nodes[nNode].threshold ? m_nodes[nNode].left : m_nodes[nNode].right;

			return m_values.data() + nNode * m_nOutputs;
		}

		// ---- CDecisionTreeClassifier ----

		CDecisionTreeClassifier::CDecisionTreeClassifier(const SDecisionTreeOptions& options) : m_options(options)
		{}

		void CDecisionTreeClassifier::Fit(const CMatrix& x, const CVectorI& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Decision tree");

			const size_t nClasses = detail::CountClasses(y, "Decision tree");

			std::vector<double> targets(y.Size());
			std::vector<size_t> samples(y.Size());
			for (size_t i = 0; i < y.Size(); ++i)
			{
				targets[i] = double(y[i]);
				samples[i] = i;
			}

			m_bFitted = false;

			std::mt19937 rng(m_options.seed);
			m_tree.Build(x, targets, nClasses, samples, m_options, rng);

			m_nFeatures = x.Cols();
			m_bFitted = true;
		}

		CVectorI CDecisionTreeClassifier::Predict(const CMatrix& x) const
		{
			detail::CheckPredict(x.Cols(), m_nFeatures, m_bFitted, "Decision tree");

			CVectorI result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
				result[i] = int(detail::ArgMax(m_tree.Predict(x.Data() + i * x.Cols()), m_tree.GetOutputCount()));

			return result;
		}

		CMatrix CDecisionTreeClassifier::PredictProbability(const CMatrix& x) const
		{
			detail::CheckPredict(x.Cols(), m_nFeatures, m_bFitted, "Decision tree");

			const size_t nClasses = m_tree.GetOutputCount();

			CMatrix result(x.Rows(), nClasses, x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
			{
				const double* pShares = m_tree.Predict(x.Data() + i * x.Cols());
				for (size_t k = 0; k < nClasses; ++k)
					result(i, k) = pShares[k];
			}

			return result;
		}

		// ---- TDecisionTreeRegressor ----

		template<class T>
		TDecisionTreeRegressor<T>::TDecisionTreeRegressor(const SDecisionTreeOptions& options) : m_options(options)
		{}

		template<class T>
		void TDecisionTreeRegressor<T>::Fit(const TMatrix<T>& x, const TVector<T>& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Decision tree");

			std::vector<T> targets(y.Data(), y.Data() + y.Size());
			std::vector<size_t> samples(y.Size());
			std::iota(samples.begin(), samples.end(), size_t(0));

			m_bFitted = false;

			std::mt19937 rng(m_options.seed);
			m_tree.Build(x, targets, 0, samples, m_options, rng);

			m_nFeatures = x.Cols();
			m_bFitted = true;
		}

		template<class T>
		TVector<T> TDecisionTreeRegressor<T>::Predict(const TMatrix<T>& x) const
		{
			detail::CheckPredict(x.Cols(), m_nFeatures, m_bFitted, "Decision tree");

			TVector<T> result(x.Rows(), x.GetAlloc());
			for (size_t i = 0; i < x.Rows(); ++i)
				result[i] = *m_tree.Predict(x.Data() + i * x.Cols());

			return result;
		}

		// The element types the library supports.
		template class TTree<float>;
		template class TTree<double>;
		template class TDecisionTreeRegressor<float>;
		template class TDecisionTreeRegressor<double>;
	}
}