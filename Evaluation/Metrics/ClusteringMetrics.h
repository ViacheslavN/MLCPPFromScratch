#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include "MetricsUtils.h"

namespace ML
{
	namespace evaluation
	{
		// How good a clustering is, without knowing the true groups. Static methods only.
		//
		// The methods are templates: 'x' is the samples (one per row) of any element
		// type, 'labels' the cluster of every sample, of any element type; both are
		// deduced from the arguments. Distances are Euclidean, the results doubles.
		// Different sizes and empty data throw CommonLib::CExcBase.
		class CClusteringMetrics
		{
		public:
			// ---- Inertia ----

			// The sum of the squared distances from every sample to the center of its
			// cluster: what k-means minimizes. Smaller is tighter, but it always
			// shrinks as clusters are added, so compare it only for the same number of
			// clusters (or look for the "elbow" as the number grows).

			// The centers are the means of the clusters; the labels can be any values.
			template<class T, class TLabel>
			static double Inertia(const numerics::TMatrix<T>& x, const numerics::TVector<TLabel>& labels)
			{
				detail::CheckPair(x.Rows(), labels.Size(), "Inertia");

				std::vector<size_t> counts;
				const std::vector<size_t> clusters = Clusters(labels, counts);
				const size_t nCols = x.Cols();

				std::vector<double> centers(counts.size() * nCols, 0.0);
				for (size_t i = 0; i < x.Rows(); ++i)
				{
					for (size_t j = 0; j < nCols; ++j)
						centers[clusters[i] * nCols + j] += double(x(i, j)) / double(counts[clusters[i]]);
				}

				double sum = 0;
				for (size_t i = 0; i < x.Rows(); ++i)
				{
					for (size_t j = 0; j < nCols; ++j)
					{
						const double difference = double(x(i, j)) - centers[clusters[i] * nCols + j];
						sum += difference * difference;
					}
				}

				return sum;
			}

			// With given centers (one per row): the label of a sample is the row of its center.
			template<class T, class TLabel>
			static double Inertia(const numerics::TMatrix<T>& x, const numerics::TVector<TLabel>& labels, const numerics::TMatrix<T>& centers)
			{
				detail::CheckPair(x.Rows(), labels.Size(), "Inertia");

				if (x.Cols() != centers.Cols())
					detail::Throw("Inertia", "the samples and the centers have different numbers of features");

				double sum = 0;
				for (size_t i = 0; i < x.Rows(); ++i)
				{
					const size_t nCenter = detail::ClassNumber(double(labels[i]), centers.Rows(), "Inertia");
					for (size_t j = 0; j < x.Cols(); ++j)
					{
						const double difference = double(x(i, j)) - double(centers(nCenter, j));
						sum += difference * difference;
					}
				}

				return sum;
			}

			// ---- Silhouette ----

			// For one sample: a = its mean distance to the other samples of its cluster,
			// b = its mean distance to the samples of the nearest other cluster, and
			//   silhouette = (b - a) / max(a, b),
			// from -1 to 1. Near 1: well inside its cluster; near 0: on the border of
			// two clusters; negative: probably in the wrong cluster. A sample alone in
			// its cluster gets 0.
			//
			// Needs at least 2 clusters and fewer clusters than samples. Compares every
			// pair of samples: n * n distances for n samples.

			// The silhouette of every sample.
			template<class T, class TLabel>
			static numerics::CVector SilhouetteSamples(const numerics::TMatrix<T>& x, const numerics::TVector<TLabel>& labels)
			{
				detail::CheckPair(x.Rows(), labels.Size(), "Silhouette");

				std::vector<size_t> counts;
				const std::vector<size_t> clusters = Clusters(labels, counts);
				const size_t nRows = x.Rows();
				const size_t nCols = x.Cols();

				if (counts.size() < 2 || counts.size() >= nRows)
					detail::Throw("Silhouette", "the number of clusters must be at least 2 and less than the number of samples");

				numerics::CVector result(nRows);
				std::vector<double> sums(counts.size());	// the distances from a sample to every cluster

				for (size_t i = 0; i < nRows; ++i)
				{
					std::fill(sums.begin(), sums.end(), 0.0);
					for (size_t k = 0; k < nRows; ++k)
					{
						double distance = 0;
						for (size_t j = 0; j < nCols; ++j)
						{
							const double difference = double(x(i, j)) - double(x(k, j));
							distance += difference * difference;
						}

						sums[clusters[k]] += std::sqrt(distance);
					}

					const size_t nOwn = clusters[i];
					if (counts[nOwn] == 1)
						continue;	// alone in its cluster: 0

					// the sample itself is in the sum with distance 0, but not in the mean
					const double own = sums[nOwn] / double(counts[nOwn] - 1);

					double nearest = std::numeric_limits<double>::infinity();
					for (size_t c = 0; c < counts.size(); ++c)
					{
						if (c != nOwn && sums[c] / double(counts[c]) < nearest)
							nearest = sums[c] / double(counts[c]);
					}

					const double larger = own > nearest ? own : nearest;
					result[i] = larger > 0 ? (nearest - own) / larger : 0.0;
				}

				return result;
			}

			// The mean silhouette of all samples: one number for the whole clustering.
			// Unlike the inertia it can be compared between different numbers of clusters.
			template<class T, class TLabel>
			static double SilhouetteScore(const numerics::TMatrix<T>& x, const numerics::TVector<TLabel>& labels)
			{
				return SilhouetteSamples(x, labels).Mean();
			}

		private:
			// The cluster number (0, 1, ...) of every sample: equal labels get the same
			// number. 'counts' receives the number of samples of every cluster.
			template<class TLabel>
			static std::vector<size_t> Clusters(const numerics::TVector<TLabel>& labels, std::vector<size_t>& counts)
			{
				std::map<TLabel, size_t> numbers;
				std::vector<size_t> clusters(labels.Size());
				for (size_t i = 0; i < labels.Size(); ++i)
					clusters[i] = numbers.emplace(labels[i], numbers.size()).first->second;

				counts.assign(numbers.size(), 0);
				for (size_t nCluster : clusters)
					++counts[nCluster];

				return clusters;
			}
		};
	}
}
