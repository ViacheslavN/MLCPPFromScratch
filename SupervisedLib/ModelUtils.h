#pragma once

// Internal helpers of SupervisedLib; not part of its interface.

#include <cstddef>

#include "Model.h"

namespace ML
{
	namespace supervised
	{
		namespace detail
		{
			// Fit: the data (nRows x nCols) has at least one row and one column, and one
			// target per row.
			void CheckFit(size_t nRows, size_t nCols, size_t nTargets, const char* pszModel);
			// Predict: the model is fitted and the data has the number of features (nCols)
			// it was fitted with.
			void CheckPredict(size_t nCols, size_t nFeatures, bool bFitted, const char* pszModel);
			// The number of classes in 'y': the largest label plus one. Throws on a
			// negative label and when there are fewer than two classes.
			size_t CountClasses(const CVectorI& y, const char* pszModel);
			// Index of the largest of 'nCount' values, the first one on a tie.
			size_t ArgMax(const double* pValues, size_t nCount);
		}
	}
}
