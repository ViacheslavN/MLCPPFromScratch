#pragma once

// Shared includes and helpers for the Evaluation Catch2 tests: the helpers of the
// NumericsCore tests (MakeMatrix, MakeVector, MaxDiff, ...) are reused.

#include "NumericsCore/Test/catch/TestCommon.h"

#include "Evaluation/Metrics/RegressionMetrics.h"
#include "Evaluation/Metrics/ClassificationMetrics.h"
#include "Evaluation/Metrics/ClusteringMetrics.h"

namespace test_utils
{
	// Exact results are compared with this margin: the metrics compute in double.
	const double c_margin = 1e-12;

	inline Catch::Approx Near(double value)
	{
		return Catch::Approx(value).margin(c_margin);
	}
}
