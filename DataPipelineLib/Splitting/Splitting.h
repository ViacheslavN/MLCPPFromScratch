#pragma once

#include <cstddef>
#include <map>
#include <vector>

#include "NumericsCoreLib/Matrix.h"
#include "NumericsCoreLib/Vector.h"

namespace ML
{
	namespace data
	{
		// ---- Options ----

		struct SSplitOptions
		{
			// The shares of the samples that go to the test and the validation sets;
			// the rest is the training set. Each is in [0, 1), their sum below 1. The
			// number of samples of a set is its share of the total, rounded.
			double testShare = 0.2;
			double validationShare = 0;
			// True: the samples are assigned at random. False: in their order, the
			// training set first, then validation, then test (for data ordered in time).
			bool shuffle = true;
			// True: every class is split in the given shares, so the sets keep the class
			// proportions of the data. The targets are the class labels.
			bool stratified = false;
			// Seed of the random numbers: the same seed gives the same split.
			unsigned int seed = 0;
		};

		struct SKFoldOptions
		{
			// The number of folds, at least 2 and at most the number of samples.
			size_t folds = 5;
			// True: the samples are assigned to the folds at random; false: in their order.
			bool shuffle = true;
			// True: every fold keeps the class proportions of the data.
			bool stratified = false;
			unsigned int seed = 0;
		};

		// ---- Results as sample indexes ----

		struct SSplitIndexes
		{
			std::vector<size_t> train;
			std::vector<size_t> validation;
			std::vector<size_t> test;
		};

		// One round of k-fold cross-validation.
		struct SFoldIndexes
		{
			std::vector<size_t> train;	// the other k - 1 folds
			std::vector<size_t> test;	// this fold
		};

		struct SBootstrapIndexes
		{
			std::vector<size_t> sample;		// drawn with repetition, in the order drawn
			std::vector<size_t> outOfBag;	// the samples that were not drawn, ascending
		};

		// ---- Results as data ----

		// Samples and their targets. TX is the element type of the features, TY of the
		// targets: for example double and double for regression, double and int for
		// classification.
		template<class TX, class TY>
		struct TDataset
		{
			numerics::TMatrix<TX> x;	// one row per sample
			numerics::TVector<TY> y;	// one target per sample
		};

		template<class TX, class TY>
		struct TSplit
		{
			TDataset<TX, TY> train;
			TDataset<TX, TY> validation;	// empty when validationShare is 0
			TDataset<TX, TY> test;
		};

		template<class TX, class TY>
		struct TFold
		{
			TDataset<TX, TY> train;
			TDataset<TX, TY> test;
		};

		template<class TX, class TY>
		struct TBootstrap
		{
			TDataset<TX, TY> sample;
			TDataset<TX, TY> outOfBag;
		};

		// Splitting data into parts: shuffle, train/validation/test split, k-fold
		// cross-validation and bootstrap. Static methods only.
		//
		// Every operation exists in two forms. The ...Indexes methods return the numbers
		// of the samples of each part and work for any data. The template methods take
		// the samples 'x' and the targets 'y' and return the parts as new matrices and
		// vectors; the element types are deduced from the arguments, and the results use
		// the allocators of 'x' and 'y'.
		//
		// Wrong sizes and options throw CommonLib::CExcBase.
		class CSplitting
		{
		public:
			// ---- Shuffle ----

			// The numbers 0..nCount-1 in a random order.
			static std::vector<size_t> ShuffleIndexes(size_t nCount, unsigned int seed = 0);

			// Puts the samples in a random order; a row of 'x' and its target stay together.
			template<class TX, class TY>
			static void Shuffle(numerics::TMatrix<TX>& x, numerics::TVector<TY>& y, unsigned int seed = 0)
			{
				TDataset<TX, TY> shuffled = Select(x, y, ShuffleIndexes(x.Rows(), seed));
				x.Swap(shuffled.x);
				y.Swap(shuffled.y);
			}

			// ---- Train / validation / test split ----

			// Not stratified: options.stratified must be false.
			static SSplitIndexes SplitIndexes(size_t nCount, const SSplitOptions& options = SSplitOptions());
			// Stratified by 'classes', the class number of every sample (any numbers;
			// equal numbers are one class). options.stratified is ignored.
			static SSplitIndexes SplitIndexes(const std::vector<size_t>& classes, const SSplitOptions& options = SSplitOptions());

			template<class TX, class TY>
			static TSplit<TX, TY> Split(const numerics::TMatrix<TX>& x, const numerics::TVector<TY>& y,
				const SSplitOptions& options = SSplitOptions())
			{
				CheckData(x.Rows(), y.Size());

				const SSplitIndexes indexes = options.stratified ? SplitIndexes(Classes(y), options) : SplitIndexes(x.Rows(), options);

				TSplit<TX, TY> result;
				result.train = Select(x, y, indexes.train);
				result.validation = Select(x, y, indexes.validation);
				result.test = Select(x, y, indexes.test);
				return result;
			}

			// ---- K-fold cross-validation ----

			// The samples are cut into 'folds' parts of equal size (the first ones get
			// one more sample when it does not divide). Round i tests on part i and
			// trains on the others, so every sample is tested exactly once.

			// Not stratified: options.stratified must be false.
			static std::vector<SFoldIndexes> KFoldIndexes(size_t nCount, const SKFoldOptions& options = SKFoldOptions());
			// Stratified by 'classes'; options.stratified is ignored.
			static std::vector<SFoldIndexes> KFoldIndexes(const std::vector<size_t>& classes, const SKFoldOptions& options = SKFoldOptions());

			template<class TX, class TY>
			static std::vector<TFold<TX, TY>> KFold(const numerics::TMatrix<TX>& x, const numerics::TVector<TY>& y,
				const SKFoldOptions& options = SKFoldOptions())
			{
				CheckData(x.Rows(), y.Size());

				const std::vector<SFoldIndexes> indexes = options.stratified ? KFoldIndexes(Classes(y), options) : KFoldIndexes(x.Rows(), options);

				std::vector<TFold<TX, TY>> result(indexes.size());
				for (size_t i = 0; i < indexes.size(); ++i)
				{
					result[i].train = Select(x, y, indexes[i].train);
					result[i].test = Select(x, y, indexes[i].test);
				}

				return result;
			}

			// ---- Bootstrap ----

			// Draws 'nSamples' of the nCount samples at random, with repetition
			// (nSamples == 0: as many as there are, nCount). About 63% of the samples
			// get drawn; the rest, the out-of-bag samples, can serve as a test set.
			static SBootstrapIndexes BootstrapIndexes(size_t nCount, unsigned int seed = 0, size_t nSamples = 0);

			template<class TX, class TY>
			static TBootstrap<TX, TY> Bootstrap(const numerics::TMatrix<TX>& x, const numerics::TVector<TY>& y,
				unsigned int seed = 0, size_t nSamples = 0)
			{
				CheckData(x.Rows(), y.Size());

				const SBootstrapIndexes indexes = BootstrapIndexes(x.Rows(), seed, nSamples);

				TBootstrap<TX, TY> result;
				result.sample = Select(x, y, indexes.sample);
				result.outOfBag = Select(x, y, indexes.outOfBag);
				return result;
			}

			// ---- Picking samples by their indexes ----

			// The rows 'indexes' of 'x', in that order; an index can repeat.
			template<class T>
			static numerics::TMatrix<T> SelectRows(const numerics::TMatrix<T>& x, const std::vector<size_t>& indexes)
			{
				CheckIndexes(indexes, x.Rows());

				const size_t nCols = x.Cols();

				numerics::TMatrix<T> result(indexes.size(), nCols, x.GetAlloc());
				for (size_t i = 0; i < indexes.size(); ++i)
				{
					const T* pSource = x.Data() + indexes[i] * nCols;
					T* pResult = result.Data() + i * nCols;
					for (size_t j = 0; j < nCols; ++j)
						pResult[j] = pSource[j];
				}

				return result;
			}

			// The elements 'indexes' of 'y', in that order.
			template<class T>
			static numerics::TVector<T> SelectElements(const numerics::TVector<T>& y, const std::vector<size_t>& indexes)
			{
				CheckIndexes(indexes, y.Size());

				numerics::TVector<T> result(indexes.size(), y.GetAlloc());
				for (size_t i = 0; i < indexes.size(); ++i)
					result[i] = y[indexes[i]];

				return result;
			}

			// The samples 'indexes' with their targets.
			template<class TX, class TY>
			static TDataset<TX, TY> Select(const numerics::TMatrix<TX>& x, const numerics::TVector<TY>& y, const std::vector<size_t>& indexes)
			{
				CheckData(x.Rows(), y.Size());

				TDataset<TX, TY> result;
				result.x = SelectRows(x, indexes);
				result.y = SelectElements(y, indexes);
				return result;
			}

			// The class number of every sample for the stratified methods: equal targets
			// get the same number.
			template<class TY>
			static std::vector<size_t> Classes(const numerics::TVector<TY>& y)
			{
				std::map<TY, size_t> numbers;
				std::vector<size_t> classes(y.Size());
				for (size_t i = 0; i < y.Size(); ++i)
					classes[i] = numbers.emplace(y[i], numbers.size()).first->second;

				return classes;
			}

		private:
			// One target per row.
			static void CheckData(size_t nRows, size_t nTargets);
			// Every index is below nCount.
			static void CheckIndexes(const std::vector<size_t>& indexes, size_t nCount);
		};
	}
}
