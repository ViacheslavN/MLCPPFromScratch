#include "TestCommon.h"

using namespace ML::numerics;
using namespace test_utils;

// ---------------------------------------------------------------------------
// Construction, copy, move
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Vector construction", "[vector]", int, float, double)
{
	typedef TVector<TestType> TVec;
	IAllocPtr alloc = MakeAlloc();

	{
		TVec empty;
		REQUIRE(empty.Empty());
		REQUIRE(empty.Size() == 0);
		REQUIRE(empty.Data() == nullptr);
		REQUIRE(empty.GetAlloc() != nullptr);	// a default allocator is created

		TVec zeros(4, alloc);
		REQUIRE(zeros.Size() == 4);
		REQUIRE(zeros.GetAlloc() == alloc);
		REQUIRE(Equals(zeros, {0, 0, 0, 0}));

		TVec filled(3, TestType(7), alloc);
		REQUIRE(Equals(filled, {7, 7, 7}));

		// a literal 0 is the fill value, not a null allocator
		TVec zeroValue(5, 0);
		REQUIRE(zeroValue.Size() == 5);

		TVec noElements(0, alloc);
		REQUIRE(noElements.Empty());
		REQUIRE(noElements.Data() == nullptr);

		REQUIRE_THROWS_AS(TVec(std::numeric_limits<size_t>::max(), alloc), CExcBase);
	}

	RequireNoLeaks(alloc);
}

TEMPLATE_TEST_CASE("Vector copy and move", "[vector]", int, float, double)
{
	typedef TVector<TestType> TVec;
	IAllocPtr alloc = MakeAlloc();

	{
		TVec source = MakeVector<TestType>({1, 2, 3}, alloc);

		TVec copy(source);
		REQUIRE(Equals(copy, {1, 2, 3}));
		REQUIRE(copy.Data() != source.Data());		// its own buffer
		REQUIRE(copy.GetAlloc() == alloc);			// the same allocator

		copy[0] = 9;
		REQUIRE(source[0] == 1);

		const TestType* pBuffer = copy.Data();
		TVec moved(std::move(copy));
		REQUIRE(Equals(moved, {9, 2, 3}));
		REQUIRE(moved.Data() == pBuffer);			// the buffer moved, not copied
		REQUIRE(copy.Empty());
		REQUIRE(copy.Data() == nullptr);

		// the moved-from vector is still usable
		copy.Resize(2);
		REQUIRE(Equals(copy, {0, 0}));

		TVec assigned;
		assigned = moved;
		REQUIRE(Equals(assigned, {9, 2, 3}));
		REQUIRE(assigned.GetAlloc() == alloc);

		TVec moveAssigned(7, alloc);
		moveAssigned = std::move(moved);
		REQUIRE(Equals(moveAssigned, {9, 2, 3}));
		REQUIRE(moved.Empty());

		TVec& self = assigned;
		assigned = self;
		REQUIRE(Equals(assigned, {9, 2, 3}));
	}

	RequireNoLeaks(alloc);
}

// ---------------------------------------------------------------------------
// Access and resizing
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Vector access, Fill, Resize, Clear, Swap", "[vector]", int, float, double)
{
	typedef TVector<TestType> TVec;
	IAllocPtr alloc = MakeAlloc();

	{
		TVec vec(3, alloc);
		vec[1] = 5;
		vec.At(2) = 6;
		REQUIRE(vec.Data()[1] == 5);
		REQUIRE(vec.At(1) == 5);
		REQUIRE(vec[2] == 6);

		const TVec& constVec = vec;
		REQUIRE(constVec[1] == 5);
		REQUIRE(constVec.At(2) == 6);
		REQUIRE_THROWS_AS(vec.At(3), CExcBase);
		REQUIRE_THROWS_AS(constVec.At(3), CExcBase);

		vec.Fill(2);
		REQUIRE(Equals(vec, {2, 2, 2}));

		vec.Resize(5);	// the old content is not kept
		REQUIRE(Equals(vec, {0, 0, 0, 0, 0}));
		REQUIRE(vec.GetAlloc() == alloc);

		vec.Clear();
		REQUIRE(vec.Empty());
		REQUIRE(vec.Data() == nullptr);
		vec.Clear();	// twice is fine

		TVec a = MakeVector<TestType>({1, 2}, alloc);
		TVec b = MakeVector<TestType>({3, 4, 5});
		a.Swap(b);
		REQUIRE(Equals(a, {3, 4, 5}));
		REQUIRE(Equals(b, {1, 2}));
		REQUIRE(b.GetAlloc() == alloc);		// the allocators are swapped with the buffers
		REQUIRE(a.GetAlloc() != alloc);
	}

	RequireNoLeaks(alloc);
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Vector element-wise operators", "[vector][operators]", int, float, double)
{
	typedef TVector<TestType> TVec;
	IAllocPtr alloc = MakeAlloc();

	{
		const TVec a = MakeVector<TestType>({1, 2, 3}, alloc);
		const TVec b = MakeVector<TestType>({4, -5, 6}, alloc);

		REQUIRE(Equals(a + b, {5, -3, 9}));
		REQUIRE(Equals(a - b, {-3, 7, -3}));
		REQUIRE(Equals(a * TestType(2), {2, 4, 6}));
		REQUIRE(Equals(TestType(2) * a, {2, 4, 6}));
		REQUIRE((a + b).GetAlloc() == alloc);	// the allocator of the left operand
		REQUIRE(Equals(a, {1, 2, 3}));			// the operands are unchanged

		TVec c(a);
		c += b;
		REQUIRE(Equals(c, {5, -3, 9}));
		c -= a;
		REQUIRE(Equals(c, {4, -5, 6}));
		c *= TestType(-1);
		REQUIRE(Equals(c, {-4, 5, -6}));
		c += c;
		REQUIRE(Equals(c, {-8, 10, -12}));

		const TVec other(2, alloc);
		REQUIRE_THROWS_AS(a + other, CExcBase);
		REQUIRE_THROWS_AS(a - other, CExcBase);
		REQUIRE_THROWS_AS(c += other, CExcBase);
		REQUIRE_THROWS_AS(c -= other, CExcBase);
		REQUIRE(Equals(c, {-8, 10, -12}));		// unchanged after a failed operation
	}

	RequireNoLeaks(alloc);
}

TEMPLATE_TEST_CASE("Vector dot product", "[vector]", int, float, double)
{
	typedef TVector<TestType> TVec;

	REQUIRE(TVec::Dot(MakeVector<TestType>({1, 2, 3}), MakeVector<TestType>({4, -5, 6})) == TestType(12));
	REQUIRE(TVec::Dot(MakeVector<TestType>({3, 4}), MakeVector<TestType>({3, 4})) == TestType(25));
	REQUIRE(TVec::Dot(MakeVector<TestType>({1, 0}), MakeVector<TestType>({0, 1})) == TestType(0));	// orthogonal
	REQUIRE(TVec::Dot(TVec(), TVec()) == TestType(0));
	REQUIRE_THROWS_AS(TVec::Dot(TVec(3), TVec(2)), CExcBase);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Vector reductions", "[vector][reductions]", int, float, double)
{
	typedef TVector<TestType> TVec;

	const TVec vec = MakeVector<TestType>({1, 7, -3, 7});
	REQUIRE(vec.Sum() == TestType(12));
	REQUIRE(vec.Mean() == 3);
	REQUIRE(vec.Max() == TestType(7));
	REQUIRE(vec.ArgMax() == 1);		// the first of the two 7s

	const TVec negative = MakeVector<TestType>({-5, -2, -9});
	REQUIRE(negative.Max() == TestType(-2));
	REQUIRE(negative.ArgMax() == 1);

	const TVec empty;
	REQUIRE(empty.Sum() == TestType(0));
	REQUIRE_THROWS_AS(empty.Mean(), CExcBase);
	REQUIRE_THROWS_AS(empty.Max(), CExcBase);
	REQUIRE_THROWS_AS(empty.ArgMax(), CExcBase);
}

TEST_CASE("Vector mean of integers is not truncated", "[vector][reductions]")
{
	// the mean of int is double; for float and double it keeps the type
	STATIC_REQUIRE(std::is_same<decltype(CVectorI().Mean()), double>::value);
	STATIC_REQUIRE(std::is_same<decltype(CVectorF().Mean()), float>::value);
	STATIC_REQUIRE(std::is_same<decltype(CVector().Mean()), double>::value);
	STATIC_REQUIRE(std::is_same<decltype(CVectorI().Sum()), int>::value);

	REQUIRE(MakeVector<int>({1, 2}).Mean() == 1.5);
	REQUIRE(MakeVector<int>({1, 2, 2}).Mean() == Catch::Approx(5.0 / 3));
}

TEST_CASE("Vector type names", "[vector]")
{
	STATIC_REQUIRE(std::is_same<CVector, TVector<double>>::value);
	STATIC_REQUIRE(std::is_same<CVectorF, TVector<float>>::value);
	STATIC_REQUIRE(std::is_same<CVectorI, TVector<int>>::value);
}
