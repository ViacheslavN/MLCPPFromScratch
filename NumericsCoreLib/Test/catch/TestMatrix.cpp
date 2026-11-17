#include "TestCommon.h"

using namespace ML::numerics;
using namespace test_utils;

// ---------------------------------------------------------------------------
// Construction, copy, move
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Matrix construction", "[matrix]", int, float, double)
{
	typedef TMatrix<TestType> TMat;
	IAllocPtr alloc = MakeAlloc();

	{
		TMat empty;
		REQUIRE(empty.Empty());
		REQUIRE(empty.Rows() == 0);
		REQUIRE(empty.Cols() == 0);
		REQUIRE(empty.Data() == nullptr);
		REQUIRE(empty.GetAlloc() != nullptr);

		TMat zeros(3, 4, alloc);
		REQUIRE(zeros.Rows() == 3);
		REQUIRE(zeros.Cols() == 4);
		REQUIRE(zeros.Size() == 12);
		REQUIRE(zeros.GetAlloc() == alloc);
		for (size_t i = 0; i < zeros.Size(); ++i)
			REQUIRE(zeros.Data()[i] == TestType(0));

		TMat filled(2, 3, TestType(5), alloc);
		REQUIRE(Equals(filled, {5, 5, 5, 5, 5, 5}));

		TMat zeroValue(2, 3, 0);
		REQUIRE(zeroValue.Size() == 6);

		// a zero dimension keeps the other one and has no buffer
		TMat noRows(0, 5, alloc);
		REQUIRE(noRows.Empty());
		REQUIRE(noRows.Cols() == 5);
		REQUIRE(noRows.Data() == nullptr);

		REQUIRE_THROWS_AS(TMat(std::numeric_limits<size_t>::max(), 3, alloc), CExcBase);
	}

	RequireNoLeaks(alloc);
}

TEMPLATE_TEST_CASE("Matrix storage is row-major", "[matrix]", int, float, double)
{
	TMatrix<TestType> matrix = MakeMatrix<TestType>(2, 3, {1, 2, 3, 4, 5, 6});

	REQUIRE(matrix(0, 0) == TestType(1));
	REQUIRE(matrix(0, 2) == TestType(3));
	REQUIRE(matrix(1, 0) == TestType(4));
	REQUIRE(matrix(1, 2) == TestType(6));

	matrix(1, 2) = 7;
	REQUIRE(matrix.Data()[1 * 3 + 2] == TestType(7));
	REQUIRE(matrix.At(1, 2) == TestType(7));

	const TMatrix<TestType>& constMatrix = matrix;
	REQUIRE(constMatrix(1, 0) == TestType(4));
	REQUIRE(constMatrix.At(0, 1) == TestType(2));

	REQUIRE_THROWS_AS(matrix.At(2, 0), CExcBase);
	REQUIRE_THROWS_AS(matrix.At(0, 3), CExcBase);
	REQUIRE_THROWS_AS(constMatrix.At(2, 0), CExcBase);
}

TEMPLATE_TEST_CASE("Matrix copy and move", "[matrix]", int, float, double)
{
	typedef TMatrix<TestType> TMat;
	IAllocPtr alloc = MakeAlloc();

	{
		TMat source = MakeMatrix<TestType>(2, 2, {1, 2, 3, 4}, alloc);

		TMat copy(source);
		REQUIRE(Equals(copy, {1, 2, 3, 4}));
		REQUIRE(copy.Rows() == 2);
		REQUIRE(copy.Data() != source.Data());
		REQUIRE(copy.GetAlloc() == alloc);

		copy(0, 0) = 9;
		REQUIRE(source(0, 0) == TestType(1));

		const TestType* pBuffer = copy.Data();
		TMat moved(std::move(copy));
		REQUIRE(Equals(moved, {9, 2, 3, 4}));
		REQUIRE(moved.Data() == pBuffer);
		REQUIRE(copy.Empty());
		REQUIRE(copy.Rows() == 0);
		REQUIRE(copy.Data() == nullptr);

		copy.Resize(1, 2);		// the moved-from matrix is still usable
		REQUIRE(Equals(copy, {0, 0}));

		TMat assigned;
		assigned = moved;
		REQUIRE(Equals(assigned, {9, 2, 3, 4}));
		REQUIRE(assigned.Cols() == 2);

		TMat moveAssigned(5, 5, alloc);
		moveAssigned = std::move(moved);
		REQUIRE(Equals(moveAssigned, {9, 2, 3, 4}));
		REQUIRE(moved.Empty());

		TMat& self = assigned;
		assigned = self;
		REQUIRE(Equals(assigned, {9, 2, 3, 4}));

		// a copy of a matrix with a zero dimension keeps the shape
		TMat noRows(0, 3, alloc);
		TMat noRowsCopy(noRows);
		REQUIRE(noRowsCopy.Rows() == 0);
		REQUIRE(noRowsCopy.Cols() == 3);
	}

	RequireNoLeaks(alloc);
}

TEMPLATE_TEST_CASE("Matrix Fill, Resize, Clear, Swap", "[matrix]", int, float, double)
{
	typedef TMatrix<TestType> TMat;
	IAllocPtr alloc = MakeAlloc();

	{
		TMat matrix(2, 2, alloc);
		matrix.Fill(3);
		REQUIRE(Equals(matrix, {3, 3, 3, 3}));

		matrix.Resize(3, 1);	// the old content is not kept
		REQUIRE(matrix.Rows() == 3);
		REQUIRE(matrix.Cols() == 1);
		REQUIRE(Equals(matrix, {0, 0, 0}));
		REQUIRE(matrix.GetAlloc() == alloc);

		matrix.Clear();
		REQUIRE(matrix.Empty());
		REQUIRE(matrix.Rows() == 0);
		REQUIRE(matrix.Cols() == 0);
		matrix.Clear();

		TMat a = MakeMatrix<TestType>(1, 2, {1, 2}, alloc);
		TMat b = MakeMatrix<TestType>(2, 1, {3, 4});
		a.Swap(b);
		REQUIRE(a.Rows() == 2);
		REQUIRE(Equals(a, {3, 4}));
		REQUIRE(b.Cols() == 2);
		REQUIRE(Equals(b, {1, 2}));
		REQUIRE(b.GetAlloc() == alloc);
	}

	RequireNoLeaks(alloc);
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Matrix element-wise operators", "[matrix][operators]", int, float, double)
{
	typedef TMatrix<TestType> TMat;
	IAllocPtr alloc = MakeAlloc();

	{
		const TMat a = MakeMatrix<TestType>(2, 3, {1, 2, 3, 4, 5, 6}, alloc);
		const TMat b = MakeMatrix<TestType>(2, 3, {6, 5, 4, 3, 2, 1}, alloc);

		REQUIRE(Equals(a + b, {7, 7, 7, 7, 7, 7}));
		REQUIRE(Equals(a - b, {-5, -3, -1, 1, 3, 5}));
		REQUIRE(Equals(a * TestType(3), {3, 6, 9, 12, 15, 18}));
		REQUIRE(Equals(TestType(3) * a, {3, 6, 9, 12, 15, 18}));
		REQUIRE(Equals((a - b) * TestType(2) + a, {-9, -4, 1, 6, 11, 16}));

		const TMat sum = a + b;
		REQUIRE(sum.Rows() == 2);
		REQUIRE(sum.Cols() == 3);
		REQUIRE(sum.GetAlloc() == alloc);

		TMat c(a);
		c += b;
		c -= a;
		REQUIRE(Equals(c, {6, 5, 4, 3, 2, 1}));
		c *= TestType(2);
		REQUIRE(Equals(c, {12, 10, 8, 6, 4, 2}));

		// the same number of elements but another shape is still a mismatch
		const TMat other = MakeMatrix<TestType>(3, 2, {1, 2, 3, 4, 5, 6}, alloc);
		REQUIRE_THROWS_AS(a + other, CExcBase);
		REQUIRE_THROWS_AS(a - other, CExcBase);
		REQUIRE_THROWS_AS(c += other, CExcBase);
		REQUIRE_THROWS_AS(c -= other, CExcBase);
	}

	RequireNoLeaks(alloc);
}

TEMPLATE_TEST_CASE("Matrix product", "[matrix][multiply]", int, float, double)
{
	typedef TMatrix<TestType> TMat;
	IAllocPtr alloc = MakeAlloc();

	{
		const TMat a = MakeMatrix<TestType>(2, 3, {1, 2, 3, 4, 5, 6}, alloc);
		const TMat b = MakeMatrix<TestType>(3, 2, {7, 8, 9, 10, 11, 12});

		const TMat ab = TMat::Multiply(a, b);
		REQUIRE(ab.Rows() == 2);
		REQUIRE(ab.Cols() == 2);
		REQUIRE(Equals(ab, {58, 64, 139, 154}));
		REQUIRE(ab.GetAlloc() == alloc);	// the allocator of the first argument

		// not commutative
		const TMat ba = TMat::Multiply(b, a);
		REQUIRE(ba.Rows() == 3);
		REQUIRE(ba.Cols() == 3);
		REQUIRE(Equals(ba, {39, 54, 69, 49, 68, 87, 59, 82, 105}));

		REQUIRE(Equals(TMat::Multiply(a, Identity<TestType>(3)), {1, 2, 3, 4, 5, 6}));
		REQUIRE(Equals(TMat::Multiply(Identity<TestType>(2), a), {1, 2, 3, 4, 5, 6}));

		REQUIRE_THROWS_AS(TMat::Multiply(a, a), CExcBase);

		// a zero dimension gives an empty result of the right shape
		const TMat product = TMat::Multiply(TMat(0, 3), b);
		REQUIRE(product.Rows() == 0);
		REQUIRE(product.Cols() == 2);
	}

	RequireNoLeaks(alloc);
}

TEMPLATE_TEST_CASE("Matrix-vector product", "[matrix][multiply]", int, float, double)
{
	typedef TMatrix<TestType> TMat;
	IAllocPtr alloc = MakeAlloc();

	{
		const TMat a = MakeMatrix<TestType>(2, 3, {1, 2, 3, 4, 5, 6}, alloc);

		const TVector<TestType> result = TMat::Multiply(a, MakeVector<TestType>({1, 2, 3}));
		REQUIRE(Equals(result, {14, 32}));
		REQUIRE(result.GetAlloc() == alloc);

		REQUIRE_THROWS_AS(TMat::Multiply(a, MakeVector<TestType>({1, 2})), CExcBase);
	}

	RequireNoLeaks(alloc);
}

TEMPLATE_TEST_CASE("Matrix transpose", "[matrix]", int, float, double)
{
	typedef TMatrix<TestType> TMat;

	const TMat a = MakeMatrix<TestType>(2, 3, {1, 2, 3, 4, 5, 6});
	const TMat b = MakeMatrix<TestType>(3, 2, {7, 8, 9, 10, 11, 12});

	const TMat transposed = TMat::Transpose(a);
	REQUIRE(transposed.Rows() == 3);
	REQUIRE(transposed.Cols() == 2);
	REQUIRE(Equals(transposed, {1, 4, 2, 5, 3, 6}));
	REQUIRE(Equals(TMat::Transpose(transposed), {1, 2, 3, 4, 5, 6}));

	// (A * B)^T = B^T * A^T
	REQUIRE(MaxDiff(TMat::Transpose(TMat::Multiply(a, b)), TMat::Multiply(TMat::Transpose(b), TMat::Transpose(a))) == 0);

	const TMat noRows = TMat::Transpose(TMat(0, 3));
	REQUIRE(noRows.Rows() == 3);
	REQUIRE(noRows.Cols() == 0);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Matrix reductions over all elements", "[matrix][reductions]", int, float, double)
{
	typedef TMatrix<TestType> TMat;

	const TMat matrix = MakeMatrix<TestType>(2, 3, {1, 5, 3, 4, 2, 6});
	REQUIRE(matrix.Sum() == TestType(21));
	REQUIRE(matrix.Mean() == 3.5);
	REQUIRE(matrix.Max() == TestType(6));

	REQUIRE(MakeMatrix<TestType>(2, 2, {-4, -1, -3, -2}).Max() == TestType(-1));

	const TMat empty(0, 3);
	REQUIRE(empty.Sum() == TestType(0));
	REQUIRE_THROWS_AS(empty.Mean(), CExcBase);
	REQUIRE_THROWS_AS(empty.Max(), CExcBase);
}

TEMPLATE_TEST_CASE("Matrix reductions along an axis", "[matrix][reductions]", int, float, double)
{
	typedef TMatrix<TestType> TMat;
	typedef std::vector<size_t> TIndexes;
	IAllocPtr alloc = MakeAlloc();

	{
		// axisRows: one result per column; axisCols: one result per row
		const TMat matrix = MakeMatrix<TestType>(2, 3, {1, 5, 3, 4, 2, 6}, alloc);

		REQUIRE(Equals(matrix.Sum(TMat::axisRows), {5, 7, 9}));
		REQUIRE(Equals(matrix.Sum(TMat::axisCols), {9, 12}));
		REQUIRE(MaxDiff(matrix.Mean(TMat::axisRows), {2.5, 3.5, 4.5}) == 0);
		REQUIRE(MaxDiff(matrix.Mean(TMat::axisCols), {3, 4}) == 0);
		REQUIRE(Equals(matrix.Max(TMat::axisRows), {4, 5, 6}));
		REQUIRE(Equals(matrix.Max(TMat::axisCols), {5, 6}));
		REQUIRE(matrix.ArgMax(TMat::axisRows) == TIndexes({1, 0, 1}));
		REQUIRE(matrix.ArgMax(TMat::axisCols) == TIndexes({1, 2}));

		REQUIRE(matrix.Sum(TMat::axisRows).GetAlloc() == alloc);
		REQUIRE(matrix.Mean(TMat::axisCols).GetAlloc() == alloc);
		REQUIRE(matrix.Max(TMat::axisCols).GetAlloc() == alloc);

		// negative values and ties: the first largest element wins
		const TMat ties = MakeMatrix<TestType>(3, 2, {-1, -2, -1, -9, -4, -2});
		REQUIRE(ties.ArgMax(TMat::axisRows) == TIndexes({0, 0}));
		REQUIRE(ties.ArgMax(TMat::axisCols) == TIndexes({0, 0, 1}));
		REQUIRE(Equals(ties.Max(TMat::axisRows), {-1, -2}));
		REQUIRE(Equals(ties.Max(TMat::axisCols), {-1, -1, -2}));
	}

	RequireNoLeaks(alloc);
}

TEMPLATE_TEST_CASE("Matrix reductions of empty matrices", "[matrix][reductions]", int, float, double)
{
	typedef TMatrix<TestType> TMat;

	// 0 x 3: nothing to reduce down the rows, three empty rows across the columns
	const TMat noRows(0, 3);
	REQUIRE(Equals(noRows.Sum(TMat::axisRows), {0, 0, 0}));
	REQUIRE(noRows.Sum(TMat::axisCols).Empty());
	REQUIRE_THROWS_AS(noRows.Mean(TMat::axisRows), CExcBase);
	REQUIRE_THROWS_AS(noRows.Max(TMat::axisRows), CExcBase);
	REQUIRE_THROWS_AS(noRows.ArgMax(TMat::axisRows), CExcBase);
	REQUIRE(noRows.Mean(TMat::axisCols).Empty());
	REQUIRE(noRows.Max(TMat::axisCols).Empty());
	REQUIRE(noRows.ArgMax(TMat::axisCols).empty());

	const TMat noCols(3, 0);
	REQUIRE(Equals(noCols.Sum(TMat::axisCols), {0, 0, 0}));
	REQUIRE_THROWS_AS(noCols.Max(TMat::axisCols), CExcBase);
	REQUIRE_THROWS_AS(noCols.ArgMax(TMat::axisCols), CExcBase);
	REQUIRE(noCols.ArgMax(TMat::axisRows).empty());
}

TEST_CASE("Matrix mean of integers is not truncated", "[matrix][reductions]")
{
	STATIC_REQUIRE(std::is_same<decltype(CMatrixI().Mean()), double>::value);
	STATIC_REQUIRE(std::is_same<decltype(CMatrixF().Mean()), float>::value);
	STATIC_REQUIRE(std::is_same<decltype(CMatrixI().Mean(CMatrixI::axisRows)), CVector>::value);
	STATIC_REQUIRE(std::is_same<decltype(CMatrixF().Mean(CMatrixF::axisRows)), CVectorF>::value);

	const CMatrixI matrix = MakeMatrix<int>(2, 2, {1, 2, 2, 2});
	REQUIRE(matrix.Mean() == 1.75);
	REQUIRE(MaxDiff(matrix.Mean(CMatrixI::axisRows), {1.5, 2}) == 0);
}

TEST_CASE("Matrix type names", "[matrix]")
{
	STATIC_REQUIRE(std::is_same<CMatrix, TMatrix<double>>::value);
	STATIC_REQUIRE(std::is_same<CMatrixF, TMatrix<float>>::value);
	STATIC_REQUIRE(std::is_same<CMatrixI, TMatrix<int>>::value);
}
