#pragma once

// Shared includes and helpers for the EvaluationLib Catch2 tests: the helpers of the
// NumericsCoreLib tests (MakeMatrix, MakeVector, MaxDiff, ...) are reused.

#include "NumericsCoreLib/Test/catch/TestCommon.h"

#include "EvaluationLib/Metrics/RegressionMetrics.h"
#include "EvaluationLib/Metrics/ClassificationMetrics.h"
#include "EvaluationLib/Metrics/ClusteringMetrics.h"

namespace test_utils
{
	// Exact results are compared with this margin: the metrics compute in double.
	const double c_margin = 1e-12;

	inline Catch::Approx Near(double value)
	{
		return Catch::Approx(value).margin(c_margin);
	}
}
