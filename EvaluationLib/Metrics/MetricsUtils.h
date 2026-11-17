#pragma once

// Internal helpers of the metrics; not part of their interface. The metric classes are
// templates in headers; what throws, and what does not depend on the element type,
// is here and compiled once in Metrics.cpp.

#include <cstddef>
#include <vector>

#include "NumericsCoreLib/Matrix.h"
#include "NumericsCoreLib/Vector.h"

namespace ML
{
	namespace evaluation
	{
		namespace detail
		{
			// Two vectors that are compared element by element: the same size, not empty.
			void CheckPair(size_t nTrue, size_t nOther, const char* pszMetric);
			// A class label as a class number. Throws when the label is negative, not a
			// whole number, or (nClasses != 0) not below nClasses.
			size_t ClassNumber(double label, size_t nClasses, const char* pszMetric);
			void Throw(const char* pszMetric, const char* pszMessage);
		}
	}
}
