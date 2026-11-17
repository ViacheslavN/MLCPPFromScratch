#include "Splitting.h"

#include <numeric>
#include <random>
#include <utility>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace data
	{
		namespace
		{
			// Random order of the elements (Fisher-Yates). rng() % n instead of a std
			// distribution, whose results differ between standard libraries: the same
			// seed gives the same split everywhere.
			void ShuffleInPlace(std::vector<size_t>& values, std::mt19937& rng)
			{
				for (size_t i = values.size(); i > 1; --i)
					std::swap(values[i - 1], values[rng() % i]);
			}

			// The numbers of samples of the validation and the test sets out of nCount.
			void SplitCounts(size_t nCount, const SSplitOptions& options, size_t& nValidation, size_t& nTest)
			{
				nTest = size_t(double(nCount) * options.testShare + 0.5);
				nValidation = size_t(double(nCount) * options.validationShare + 0.5);

				// Rounding both up can take everything from a tiny set.
				while (nTest + nValidation > nCount)
				{
					if (nValidation > 0)
						--nValidation;
					else
						--nTest;
				}
			}

			void CheckSplitOptions(const SSplitOptions& options)
			{
				if (!(options.testShare >= 0 && options.testShare < 1))
					throw CommonLib::CExcBase("Split: the test share {0} must be in [0, 1)", options.testShare);

				if (!(options.validationShare >= 0 && options.validationShare < 1))
					throw CommonLib::CExcBase("Split: the validation share {0} must be in [0, 1)", options.validationShare);

				if (!(options.testShare + options.validationShare < 1))
				{
					throw CommonLib::CExcBase("Split: the test share {0} and the validation share {1} leave nothing for training",
						options.testShare, options.validationShare);
				}
			}

			void CheckFolds(size_t nCount, const SKFoldOptions& options)
			{
				if (options.folds < 2)
					throw CommonLib::CExcBase("K-fold: {0} folds, at least 2 are needed", options.folds);

				if (options.folds > nCount)
					throw CommonLib::CExcBase("K-fold: {0} folds for {1} samples", options.folds, nCount);
			}

			// The samples of every class, in their order: groups[k] lists the samples
			// whose class number is the k-th smallest.
			std::vector<std::vector<size_t>> GroupByClass(const std::vector<size_t>& classes)
			{
				std::map<size_t, size_t> numbers;
				for (size_t nClass : classes)
					numbers.emplace(nClass, 0);

				size_t nGroup = 0;
				for (auto& number : numbers)
					number.second = nGroup++;

				std::vector<std::vector<size_t>> groups(numbers.size());
				for (size_t i = 0; i < classes.size(); ++i)
					groups[numbers[classes[i]]].push_back(i);

				return groups;
			}

			// The final order of a part: random when shuffling, else the order of the data.
			void Arrange(std::vector<size_t>& indexes, bool bShuffle, std::mt19937& rng)
			{
				std::sort(indexes.begin(), indexes.end());
				if (bShuffle)
					ShuffleInPlace(indexes, rng);
			}

			// The folds as test sets -> the rounds (train = all the other folds).
			std::vector<SFoldIndexes> MakeRounds(std::vector<std::vector<size_t>>& folds)
			{
				std::vector<SFoldIndexes> rounds(folds.size());
				for (size_t i = 0; i < folds.size(); ++i)
				{
					for (size_t k = 0; k < folds.size(); ++k)
					{
						if (k != i)
							rounds[i].train.insert(rounds[i].train.end(), folds[k].begin(), folds[k].end());
					}
				}

				for (size_t i = 0; i < folds.size(); ++i)
					rounds[i].test = std::move(folds[i]);

				return rounds;
			}
		}

		// ---- Shuffle ----

		std::vector<size_t> CSplitting::ShuffleIndexes(size_t nCount, unsigned int seed)
		{
			std::vector<size_t> indexes(nCount);
			std::iota(indexes.begin(), indexes.end(), size_t(0));

			std::mt19937 rng(seed);
			ShuffleInPlace(indexes, rng);

			return indexes;
		}

		// ---- Split ----

		SSplitIndexes CSplitting::SplitIndexes(size_t nCount, const SSplitOptions& options)
		{
			CheckSplitOptions(options);

			if (options.stratified)
				throw CommonLib::CExcBase("Split: a stratified split needs the classes of the samples");

			size_t nValidation = 0;
			size_t nTest = 0;
			SplitCounts(nCount, options, nValidation, nTest);

			const size_t nTrain = nCount - nValidation - nTest;
			if (nTrain == 0 && nCount > 0)
				throw CommonLib::CExcBase("Split: {0} samples leave nothing for training", nCount);

			std::vector<size_t> indexes(nCount);
			std::iota(indexes.begin(), indexes.end(), size_t(0));

			if (options.shuffle)
			{
				std::mt19937 rng(options.seed);
				ShuffleInPlace(indexes, rng);
			}

			SSplitIndexes result;
			result.train.assign(indexes.begin(), indexes.begin() + nTrain);
			result.validation.assign(indexes.begin() + nTrain, indexes.begin() + nTrain + nValidation);
			result.test.assign(indexes.begin() + nTrain + nValidation, indexes.end());
			return result;
		}

		SSplitIndexes CSplitting::SplitIndexes(const std::vector<size_t>& classes, const SSplitOptions& options)
		{
			CheckSplitOptions(options);

			std::mt19937 rng(options.seed);

			// The plain split, applied to every class separately.
			SSplitIndexes result;
			for (std::vector<size_t>& group : GroupByClass(classes))
			{
				if (options.shuffle)
					ShuffleInPlace(group, rng);

				size_t nValidation = 0;
				size_t nTest = 0;
				SplitCounts(group.size(), options, nValidation, nTest);

				const size_t nTrain = group.size() - nValidation - nTest;
				result.train.insert(result.train.end(), group.begin(), group.begin() + nTrain);
				result.validation.insert(result.validation.end(), group.begin() + nTrain, group.begin() + nTrain + nValidation);
				result.test.insert(result.test.end(), group.begin() + nTrain + nValidation, group.end());
			}

			if (result.train.empty() && !classes.empty())
				throw CommonLib::CExcBase("Split: {0} samples leave nothing for training", classes.size());

			// The parts are grouped by class now.
			Arrange(result.train, options.shuffle, rng);
			Arrange(result.validation, options.shuffle, rng);
			Arrange(result.test, options.shuffle, rng);
			return result;
		}

		// ---- K-fold ----

		std::vector<SFoldIndexes> CSplitting::KFoldIndexes(size_t nCount, const SKFoldOptions& options)
		{
			CheckFolds(nCount, options);

			if (options.stratified)
				throw CommonLib::CExcBase("K-fold: stratified folds need the classes of the samples");

			std::vector<size_t> indexes(nCount);
			std::iota(indexes.begin(), indexes.end(), size_t(0));

			if (options.shuffle)
			{
				std::mt19937 rng(options.seed);
				ShuffleInPlace(indexes, rng);
			}

			// Consecutive pieces; the first nCount % folds pieces are one sample longer.
			std::vector<std::vector<size_t>> folds(options.folds);
			size_t nBegin = 0;
			for (size_t i = 0; i < options.folds; ++i)
			{
				const size_t nSize = nCount / options.folds + (i < nCount % options.folds ? 1 : 0);
				folds[i].assign(indexes.begin() + nBegin, indexes.begin() + nBegin + nSize);
				nBegin += nSize;
			}

			return MakeRounds(folds);
		}

		std::vector<SFoldIndexes> CSplitting::KFoldIndexes(const std::vector<size_t>& classes, const SKFoldOptions& options)
		{
			CheckFolds(classes.size(), options);

			std::mt19937 rng(options.seed);

			// The samples of every class are dealt to the folds in turn, like cards. The
			// dealing goes on from class to class, so the folds also end up equal in size.
			std::vector<std::vector<size_t>> folds(options.folds);
			size_t nNext = 0;
			for (std::vector<size_t>& group : GroupByClass(classes))
			{
				if (options.shuffle)
					ShuffleInPlace(group, rng);

				for (size_t nSample : group)
				{
					folds[nNext].push_back(nSample);
					nNext = (nNext + 1) % options.folds;
				}
			}

			for (std::vector<size_t>& fold : folds)
				Arrange(fold, options.shuffle, rng);

			std::vector<SFoldIndexes> rounds = MakeRounds(folds);
			for (SFoldIndexes& round : rounds)
				Arrange(round.train, options.shuffle, rng);

			return rounds;
		}

		// ---- Bootstrap ----

		SBootstrapIndexes CSplitting::BootstrapIndexes(size_t nCount, unsigned int seed, size_t nSamples)
		{
			SBootstrapIndexes result;
			if (nCount == 0)
				return result;

			if (nSamples == 0)
				nSamples = nCount;

			std::mt19937 rng(seed);
			std::vector<bool> drawn(nCount, false);

			result.sample.resize(nSamples);
			for (size_t i = 0; i < nSamples; ++i)
			{
				result.sample[i] = rng() % nCount;
				drawn[result.sample[i]] = true;
			}

			for (size_t i = 0; i < nCount; ++i)
			{
				if (!drawn[i])
					result.outOfBag.push_back(i);
			}

			return result;
		}

		// ---- Checks ----

		void CSplitting::CheckData(size_t nRows, size_t nTargets)
		{
			if (nRows != nTargets)
				throw CommonLib::CExcBase("Splitting: {0} samples and {1} targets", nRows, nTargets);
		}

		void CSplitting::CheckIndexes(const std::vector<size_t>& indexes, size_t nCount)
		{
			for (size_t nIndex : indexes)
			{
				if (nIndex >= nCount)
					throw CommonLib::CExcBase("Splitting: index {0} is out of range, there are {1} samples", nIndex, nCount);
			}
		}
	}
}
