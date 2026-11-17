#include "TestCommon.h"

#include "NumericsCoreLib/Decompositions.h"

using namespace ML::numerics;
using namespace test_utils;

// The decompositions are tested the same way for float and double; only the tolerance
// differs (see Tolerance()). Most checks multiply the factors back together.

// ---------------------------------------------------------------------------
// LU
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("LU reproduces the matrix: P * A = L * U", "[decompositions][lu]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	for (size_t n : {1, 2, 5, 12, 30})
	{
		const TMat a = random.Matrix<TestType>(n, n);
		const typename TDec::SLU lu = TDec::LU(a);

		REQUIRE_FALSE(lu.singular);
		REQUIRE(lu.pivots.size() == n);

		// split the combined matrix into L (unit diagonal) and U
		TMat l = Identity<TestType>(n);
		TMat u(n, n);
		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = 0; j < n; ++j)
			{
				if (j < i)
					l(i, j) = lu.lu(i, j);
				else
					u(i, j) = lu.lu(i, j);
			}
		}

		// apply the row swaps to A
		TMat permuted(a);
		for (size_t k = 0; k < n; ++k)
		{
			for (size_t j = 0; j < n && lu.pivots[k] != k; ++j)
				std::swap(permuted(k, j), permuted(lu.pivots[k], j));
		}

		REQUIRE(MaxDiff(Mul(l, u), permuted) < tolerance);
	}
}

TEMPLATE_TEST_CASE("LU solves linear systems", "[decompositions][lu]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	// needs a row swap: the first pivot is zero
	REQUIRE(MaxDiff(TDec::Solve(MakeMatrix<TestType>(2, 2, {0, 2, 3, 0}), MakeVector<TestType>({4, 9})), {3, 2}) < tolerance);

	for (size_t n : {1, 3, 10, 30})
	{
		const TMat a = random.Matrix<TestType>(n, n);
		const TVector<TestType> b = random.Vector<TestType>(n);

		const TVector<TestType> x = TDec::Solve(a, b);
		REQUIRE(MaxDiff(TMat::Multiply(a, x), b) < 100 * tolerance);

		// one decomposition, several right-hand sides
		const typename TDec::SLU lu = TDec::LU(a);
		REQUIRE(MaxDiff(TDec::SolveLU(lu, b), x) == 0);
	}

	REQUIRE_THROWS_AS(TDec::Solve(MakeMatrix<TestType>(2, 2, {1, 2, 2, 4}), MakeVector<TestType>({1, 2})), CExcBase);	// singular
	REQUIRE_THROWS_AS(TDec::Solve(Identity<TestType>(3), MakeVector<TestType>({1, 2})), CExcBase);						// wrong size
	REQUIRE_THROWS_AS(TDec::LU(MakeMatrix<TestType>(2, 3, {1, 2, 3, 4, 5, 6})), CExcBase);								// not square
}

TEMPLATE_TEST_CASE("Determinant", "[decompositions][lu]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	REQUIRE(TDec::Determinant(MakeMatrix<TestType>(2, 2, {4, 3, 6, 3})) == Catch::Approx(-6).margin(tolerance));
	REQUIRE(TDec::Determinant(MakeMatrix<TestType>(2, 2, {0, 1, 1, 0})) == Catch::Approx(-1).margin(tolerance));		// one swap: the sign
	REQUIRE(TDec::Determinant(MakeMatrix<TestType>(3, 3, {2, 0, 1, 1, 3, 2, 1, 1, 1})) == Catch::Approx(0).margin(tolerance));
	REQUIRE(TDec::Determinant(MakeMatrix<TestType>(3, 3, {2, 0, 0, 0, 3, 0, 0, 0, 4})) == Catch::Approx(24).margin(tolerance));
	REQUIRE(TDec::Determinant(MakeMatrix<TestType>(2, 2, {1, 2, 2, 4})) == 0);		// singular: exactly 0
	REQUIRE(TDec::Determinant(Identity<TestType>(4)) == 1);
	REQUIRE(TDec::Determinant(TMatrix<TestType>(0, 0)) == 1);						// the empty product

	// det(A * B) = det(A) * det(B), det(A^T) = det(A)
	const TMatrix<TestType> a = random.Matrix<TestType>(5, 5);
	const TMatrix<TestType> b = random.Matrix<TestType>(5, 5);
	const double detA = TDec::Determinant(a);
	REQUIRE(TDec::Determinant(Mul(a, b)) == Catch::Approx(detA * double(TDec::Determinant(b))).margin(tolerance));
	REQUIRE(TDec::Determinant(Transposed(a)) == Catch::Approx(detA).margin(tolerance));
}

// ---------------------------------------------------------------------------
// Cholesky
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Cholesky", "[decompositions][cholesky]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	// the textbook example
	const TMat l = TDec::Cholesky(MakeMatrix<TestType>(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98}));
	REQUIRE(MaxDiff(l, 3, 3, {2, 0, 0, 6, 1, 0, -8, 5, 3}) < tolerance);

	for (size_t n : {1, 3, 10, 25})
	{
		// B^T * B + I is symmetric positive definite
		const TMat b = random.Matrix<TestType>(n + 2, n);
		const TMat a = Mul(Transposed(b), b) + Identity<TestType>(n);

		const TMat factor = TDec::Cholesky(a);
		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = i + 1; j < n; ++j)
				REQUIRE(factor(i, j) == 0);		// lower triangular
		}

		REQUIRE(MaxDiff(Mul(factor, Transposed(factor)), a) < 10 * tolerance);

		const TVector<TestType> rhs = random.Vector<TestType>(n);
		const TVector<TestType> x = TDec::SolveCholesky(factor, rhs);
		REQUIRE(MaxDiff(TMat::Multiply(a, x), rhs) < 100 * tolerance);
		REQUIRE(MaxDiff(x, TDec::Solve(a, rhs)) < 100 * tolerance);		// agrees with LU
	}

	REQUIRE_THROWS_AS(TDec::Cholesky(MakeMatrix<TestType>(2, 2, {1, 2, 2, 1})), CExcBase);			// indefinite
	REQUIRE_THROWS_AS(TDec::Cholesky(MakeMatrix<TestType>(2, 2, {1, 1, 1, 1})), CExcBase);			// only semi-definite
	REQUIRE_THROWS_AS(TDec::Cholesky(MakeMatrix<TestType>(2, 3, {1, 2, 3, 4, 5, 6})), CExcBase);	// not square
	REQUIRE_THROWS_AS(TDec::SolveCholesky(l, MakeVector<TestType>({1, 2})), CExcBase);				// wrong size
}

// ---------------------------------------------------------------------------
// QR
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("QR", "[decompositions][qr]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	typedef std::pair<size_t, size_t> TSize;
	for (const TSize& size : {TSize(1, 1), TSize(3, 3), TSize(5, 2), TSize(20, 7), TSize(30, 30)})
	{
		const size_t m = size.first;
		const size_t n = size.second;

		const TMat a = random.Matrix<TestType>(m, n);
		const typename TDec::SQR qr = TDec::QR(a);

		REQUIRE(qr.q.Rows() == m);
		REQUIRE(qr.q.Cols() == n);
		REQUIRE(qr.r.Rows() == n);
		REQUIRE(qr.r.Cols() == n);

		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = 0; j < i; ++j)
				REQUIRE(qr.r(i, j) == 0);		// upper triangular
		}

		REQUIRE(MaxDiff(Mul(Transposed(qr.q), qr.q), Identity<TestType>(n)) < 10 * tolerance);	// orthonormal columns
		REQUIRE(MaxDiff(Mul(qr.q, qr.r), a) < 10 * tolerance);
	}

	// dependent columns and a zero matrix still decompose
	const TMat dependent = MakeMatrix<TestType>(3, 3, {1, 2, 0, 2, 4, 0, 3, 6, 0});
	const typename TDec::SQR qr = TDec::QR(dependent);
	REQUIRE(MaxDiff(Mul(qr.q, qr.r), dependent) < 10 * tolerance);

	const TMat zero(3, 2);
	const typename TDec::SQR qrZero = TDec::QR(zero);
	REQUIRE(MaxDiff(Mul(qrZero.q, qrZero.r), zero) == 0);
	REQUIRE(MaxDiff(Mul(Transposed(qrZero.q), qrZero.q), Identity<TestType>(2)) == 0);

	REQUIRE_THROWS_AS(TDec::QR(MakeMatrix<TestType>(2, 3, {1, 2, 3, 4, 5, 6})), CExcBase);	// more columns than rows
}

TEMPLATE_TEST_CASE("Least squares", "[decompositions][qr]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	// the columns are (1, x) for x = 0, 1, 2, 3
	const TMat line = MakeMatrix<TestType>(4, 2, {1, 0, 1, 1, 1, 2, 1, 3});

	// points exactly on y = 1 + 2x
	REQUIRE(MaxDiff(TDec::LeastSquares(line, MakeVector<TestType>({1, 3, 5, 7})), {1, 2}) < 10 * tolerance);
	// points (0,1) (1,2) (2,2) (3,4): the best line is y = 0.9 + 0.9x
	REQUIRE(MaxDiff(TDec::LeastSquares(line, MakeVector<TestType>({1, 2, 2, 4})), {0.9, 0.9}) < 10 * tolerance);

	// at the minimum the residual is orthogonal to every column: A^T * (A * x - b) = 0
	const TMat a = random.Matrix<TestType>(40, 5);
	const TVector<TestType> b = random.Vector<TestType>(40);
	const TVector<TestType> x = TDec::LeastSquares(a, b);
	REQUIRE(MaxDiff(TMat::Multiply(Transposed(a), TMat::Multiply(a, x) - b), TVector<TestType>(5)) < 100 * tolerance);

	// a square system has the exact solution
	const TMat square = random.Matrix<TestType>(6, 6);
	const TVector<TestType> rhs = random.Vector<TestType>(6);
	REQUIRE(MaxDiff(TDec::LeastSquares(square, rhs), TDec::Solve(square, rhs)) < 1000 * tolerance);

	REQUIRE_THROWS_AS(TDec::LeastSquares(line, MakeVector<TestType>({1, 2, 3})), CExcBase);										// wrong size
	REQUIRE_THROWS_AS(TDec::LeastSquares(MakeMatrix<TestType>(3, 2, {1, 2, 2, 4, 3, 6}), MakeVector<TestType>({1, 2, 3})), CExcBase);	// dependent columns
}

// ---------------------------------------------------------------------------
// Eigen-decomposition
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Eigen-decomposition of known matrices", "[decompositions][eigen]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	const double tolerance = Tolerance<TestType>();

	typename TDec::SEigen eigen = TDec::EigenSymmetric(MakeMatrix<TestType>(2, 2, {2, 1, 1, 2}));
	REQUIRE(MaxDiff(eigen.values, {3, 1}) < tolerance);
	// eigenvectors (1, 1) / sqrt(2) and (1, -1) / sqrt(2), up to the sign
	REQUIRE(std::fabs(double(eigen.vectors(0, 0))) == Catch::Approx(std::sqrt(0.5)).margin(tolerance));
	REQUIRE(double(eigen.vectors(0, 0)) == Catch::Approx(double(eigen.vectors(1, 0))).margin(tolerance));
	REQUIRE(double(eigen.vectors(0, 1)) == Catch::Approx(-double(eigen.vectors(1, 1))).margin(tolerance));

	eigen = TDec::EigenSymmetric(MakeMatrix<TestType>(3, 3, {2, 0, 0, 0, 3, 4, 0, 4, 9}));
	REQUIRE(MaxDiff(eigen.values, {11, 2, 1}) < 10 * tolerance);

	// a diagonal matrix: the values sorted, the vectors are unit vectors
	eigen = TDec::EigenSymmetric(MakeMatrix<TestType>(3, 3, {1, 0, 0, 0, 5, 0, 0, 0, -2}));
	REQUIRE(Equals(eigen.values, {5, 1, -2}));
	REQUIRE(eigen.vectors(1, 0) == 1);
	REQUIRE(eigen.vectors(0, 1) == 1);
	REQUIRE(eigen.vectors(2, 2) == 1);

	eigen = TDec::EigenSymmetric(TMatrix<TestType>(3, 3));		// the zero matrix
	REQUIRE(Equals(eigen.values, {0, 0, 0}));
	REQUIRE(MaxDiff(eigen.vectors, Identity<TestType>(3)) == 0);

	REQUIRE(TDec::EigenSymmetric(TMatrix<TestType>(0, 0)).values.Empty());

	REQUIRE_THROWS_AS(TDec::EigenSymmetric(MakeMatrix<TestType>(2, 2, {1, 2, 3, 4})), CExcBase);			// not symmetric
	REQUIRE_THROWS_AS(TDec::EigenSymmetric(MakeMatrix<TestType>(2, 3, {1, 2, 3, 4, 5, 6})), CExcBase);	// not square

	const double nan = std::numeric_limits<double>::quiet_NaN();
	REQUIRE_THROWS_AS(TDec::EigenSymmetric(MakeMatrix<TestType>(2, 2, {1, nan, nan, 1})), CExcBase);
}

TEMPLATE_TEST_CASE("Eigen-decomposition of random symmetric matrices", "[decompositions][eigen]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	for (size_t n : {1, 2, 6, 20, 40})
	{
		const TMat b = random.Matrix<TestType>(n, n);
		const TMat a = b + Transposed(b);

		const typename TDec::SEigen eigen = TDec::EigenSymmetric(a);

		REQUIRE(MaxDiff(Mul(Transposed(eigen.vectors), eigen.vectors), Identity<TestType>(n)) < 10 * tolerance);	// orthonormal
		REQUIRE(MaxDiff(Mul(a, eigen.vectors), Mul(eigen.vectors, Diagonal(eigen.values))) < 100 * tolerance);		// A * v = value * v
		REQUIRE(MaxDiff(Mul(Mul(eigen.vectors, Diagonal(eigen.values)), Transposed(eigen.vectors)), a) < 100 * tolerance);

		for (size_t i = 1; i < n; ++i)
			REQUIRE(eigen.values[i - 1] >= eigen.values[i]);		// the largest first

		// the sum of the eigenvalues is the trace
		double trace = 0;
		for (size_t i = 0; i < n; ++i)
			trace += double(a(i, i));

		REQUIRE(double(eigen.values.Sum()) == Catch::Approx(trace).margin(100 * tolerance));
	}
}

// ---------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("SVD", "[decompositions][svd]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	REQUIRE(MaxDiff(TDec::SVD(MakeMatrix<TestType>(2, 2, {3, 0, 0, 4})).s, {4, 3}) < tolerance);
	REQUIRE(MaxDiff(TDec::SVD(MakeMatrix<TestType>(3, 2, {1, 2, 3, 4, 5, 6})).s, {9.525518091565107, 0.514300580658644}) < 10 * tolerance);
	// the transposed matrix has the same singular values
	REQUIRE(MaxDiff(TDec::SVD(MakeMatrix<TestType>(2, 3, {1, 3, 5, 2, 4, 6})).s, {9.525518091565107, 0.514300580658644}) < 10 * tolerance);

	typedef std::pair<size_t, size_t> TSize;
	for (const TSize& size : {TSize(1, 1), TSize(4, 4), TSize(7, 4), TSize(4, 7), TSize(30, 12), TSize(12, 30), TSize(1, 5), TSize(5, 1)})
	{
		const size_t m = size.first;
		const size_t n = size.second;
		const size_t k = std::min(m, n);

		const TMat a = random.Matrix<TestType>(m, n);
		const typename TDec::SSVD svd = TDec::SVD(a);

		REQUIRE(svd.u.Rows() == m);
		REQUIRE(svd.u.Cols() == k);
		REQUIRE(svd.s.Size() == k);
		REQUIRE(svd.v.Rows() == n);
		REQUIRE(svd.v.Cols() == k);

		REQUIRE(MaxDiff(Mul(Transposed(svd.u), svd.u), Identity<TestType>(k)) < 10 * tolerance);
		REQUIRE(MaxDiff(Mul(Transposed(svd.v), svd.v), Identity<TestType>(k)) < 10 * tolerance);
		REQUIRE(MaxDiff(Mul(Mul(svd.u, Diagonal(svd.s)), Transposed(svd.v)), a) < 10 * tolerance);

		for (size_t i = 0; i < k; ++i)
		{
			REQUIRE(svd.s[i] >= 0);
			if (i > 0)
				REQUIRE(svd.s[i - 1] >= svd.s[i]);		// the largest first
		}

		// the squared singular values are the eigenvalues of A^T * A (or A * A^T)
		const typename TDec::SEigen eigen = TDec::EigenSymmetric(m >= n ? Mul(Transposed(a), a) : Mul(a, Transposed(a)));
		for (size_t i = 0; i < k; ++i)
			REQUIRE(double(svd.s[i]) * double(svd.s[i]) == Catch::Approx(double(eigen.values[i])).margin(100 * tolerance));
	}

	REQUIRE(TDec::SVD(TMat(0, 0)).s.Empty());
	REQUIRE(MaxDiff(TDec::SVD(TMat(3, 2)).s, {0, 0}) == 0);

	const double nan = std::numeric_limits<double>::quiet_NaN();
	REQUIRE_THROWS_AS(TDec::SVD(MakeMatrix<TestType>(2, 2, {1, nan, 0, 1})), CExcBase);
}

TEMPLATE_TEST_CASE("SVD of a rank-deficient matrix", "[decompositions][svd]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	// 5 x 4 of rank 2, and its transpose
	const TMat rank2 = Mul(random.Matrix<TestType>(5, 2), random.Matrix<TestType>(2, 4));
	for (const TMat& a : {rank2, Transposed(rank2)})
	{
		const typename TDec::SSVD svd = TDec::SVD(a);

		REQUIRE(svd.s[0] > 0);
		REQUIRE(svd.s[1] > 0);
		REQUIRE(svd.s[2] == 0);		// zero up to rounding is returned as exactly 0
		REQUIRE(svd.s[3] == 0);
		REQUIRE(MaxDiff(Mul(Mul(svd.u, Diagonal(svd.s)), Transposed(svd.v)), a) < 10 * tolerance);
	}
}

TEMPLATE_TEST_CASE("Pseudo-inverse", "[decompositions][svd]", float, double)
{
	typedef TDecompositions<TestType> TDec;
	typedef TMatrix<TestType> TMat;
	const double tolerance = Tolerance<TestType>();
	CRandom random;

	// for an invertible matrix it is the inverse
	REQUIRE(MaxDiff(TDec::PseudoInverse(MakeMatrix<TestType>(2, 2, {2, 0, 0, 4})), 2, 2, {0.5, 0, 0, 0.25}) < tolerance);
	REQUIRE(MaxDiff(TDec::PseudoInverse(MakeMatrix<TestType>(2, 2, {1, 2, 3, 4})), 2, 2, {-2, 1, 1.5, -0.5}) < 100 * tolerance);
	REQUIRE(MaxDiff(TDec::PseudoInverse(TMat(3, 2)), TMat(2, 3)) == 0);		// of a zero matrix: zero

	const TMat rank2 = Mul(random.Matrix<TestType>(5, 2), random.Matrix<TestType>(2, 4));
	for (const TMat& a : {random.Matrix<TestType>(4, 4), random.Matrix<TestType>(7, 4), random.Matrix<TestType>(4, 7), rank2})
	{
		const TMat inverse = TDec::PseudoInverse(a);
		REQUIRE(inverse.Rows() == a.Cols());
		REQUIRE(inverse.Cols() == a.Rows());

		// the four Moore-Penrose conditions
		REQUIRE(MaxDiff(Mul(Mul(a, inverse), a), a) < 100 * tolerance);
		REQUIRE(MaxDiff(Mul(Mul(inverse, a), inverse), inverse) < 1000 * tolerance);
		REQUIRE(MaxDiff(Transposed(Mul(a, inverse)), Mul(a, inverse)) < 100 * tolerance);
		REQUIRE(MaxDiff(Transposed(Mul(inverse, a)), Mul(inverse, a)) < 100 * tolerance);
	}

	// with independent columns, pseudo-inverse * b is the least squares solution
	const TMat tall = random.Matrix<TestType>(12, 4);
	const TVector<TestType> b = random.Vector<TestType>(12);
	REQUIRE(MaxDiff(TMat::Multiply(TDec::PseudoInverse(tall), b), TDec::LeastSquares(tall, b)) < 100 * tolerance);
}

TEST_CASE("Decompositions keep the allocator of the input", "[decompositions]")
{
	IAllocPtr alloc = MakeAlloc();

	{
		CRandom random;
		CMatrix a(4, 4, alloc);
		for (size_t i = 0; i < a.Size(); ++i)
			a.Data()[i] = random.Next();

		CVector b(4, alloc);
		const CMatrix symmetric = a + CMatrix::Transpose(a);
		CMatrix positive = CMatrix::Multiply(CMatrix::Transpose(a), a);
		for (size_t i = 0; i < 4; ++i)
			positive(i, i) += 1;

		REQUIRE(CDecompositions::LU(a).lu.GetAlloc() == alloc);
		REQUIRE(CDecompositions::Solve(a, b).GetAlloc() == alloc);
		REQUIRE(CDecompositions::Cholesky(positive).GetAlloc() == alloc);
		REQUIRE(CDecompositions::QR(a).q.GetAlloc() == alloc);
		REQUIRE(CDecompositions::QR(a).r.GetAlloc() == alloc);
		REQUIRE(CDecompositions::LeastSquares(a, b).GetAlloc() == alloc);
		REQUIRE(CDecompositions::EigenSymmetric(symmetric).values.GetAlloc() == alloc);
		REQUIRE(CDecompositions::EigenSymmetric(symmetric).vectors.GetAlloc() == alloc);
		REQUIRE(CDecompositions::SVD(a).u.GetAlloc() == alloc);
		REQUIRE(CDecompositions::SVD(a).s.GetAlloc() == alloc);
		REQUIRE(CDecompositions::SVD(a).v.GetAlloc() == alloc);
		REQUIRE(CDecompositions::PseudoInverse(a).GetAlloc() == alloc);
	}

	RequireNoLeaks(alloc);
}
