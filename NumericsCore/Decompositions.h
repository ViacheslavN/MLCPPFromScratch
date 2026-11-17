#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include "Matrix.h"
#include "Vector.h"

namespace ML
{
	namespace numerics
	{
		// Matrix decompositions and what is built on them (linear systems, determinant,
		// least squares, pseudo-inverse). Static methods only. T is float or double (the
		// member functions are compiled for these two in Decompositions.cpp). The results
		// use the allocator of the input matrix. Wrong sizes and matrices a method cannot
		// handle throw CommonLib::CExcBase.
		template<class T>
		class TDecompositions
		{
		public:
			typedef T TValue;
			typedef TMatrix<T> TMatrixType;
			typedef TVector<T> TVectorType;

			// ---- LU: square systems, determinant ----

			// P * A = L * U with partial (row) pivoting.
			struct SLU
			{
				// L and U in one matrix: U on and above the diagonal, L below it
				// (the diagonal of L is all ones and is not stored).
				TMatrixType lu;
				// Row swaps in the order they were made: at step k, row k was swapped
				// with row pivots[k].
				std::vector<size_t> pivots;
				// +1 or -1: the parity of the row swaps, the sign of the determinant.
				int sign = 1;
				// True when a pivot is exactly zero: the matrix has no inverse.
				bool singular = false;
			};

			// 'a' must be square.
			static SLU LU(const TMatrixType& a);
			// Solves A * x = b with an existing LU. Throws when the matrix is singular.
			static TVectorType SolveLU(const SLU& lu, const TVectorType& b);
			// Product of the diagonal of U with the sign of the swaps; 0 when singular.
			static TValue Determinant(const SLU& lu);

			// The same in one call: LU of 'a', then the solve or the determinant.
			static TVectorType Solve(const TMatrixType& a, const TVectorType& b);
			static TValue Determinant(const TMatrixType& a);

			// ---- Cholesky: symmetric positive definite matrices ----

			// A = L * L^T. Returns the lower triangular L. 'a' must be square, symmetric
			// and positive definite; only its lower triangle is read. Throws when the
			// matrix is not positive definite.
			static TMatrixType Cholesky(const TMatrixType& a);
			// Solves A * x = b with the L returned by Cholesky().
			static TVectorType SolveCholesky(const TMatrixType& l, const TVectorType& b);

			// ---- QR: least squares ----

			// A = Q * R for an m x n matrix with m >= n (Householder reflections).
			struct SQR
			{
				TMatrixType q;	// m x n, orthonormal columns: Q^T * Q = I
				TMatrixType r;	// n x n, upper triangular; diagonal elements can be negative
			};

			static SQR QR(const TMatrixType& a);
			// The x that minimizes |A * x - b| for an m x n matrix with m >= n.
			// Throws when the columns of 'a' are linearly dependent.
			static TVectorType LeastSquares(const TMatrixType& a, const TVectorType& b);

			// ---- Eigen-decomposition of a symmetric matrix ----

			// A = V * diag(values) * V^T (cyclic Jacobi rotations).
			struct SEigen
			{
				TVectorType values;		// eigenvalues, the largest first
				TMatrixType vectors;	// column j is the unit eigenvector of values[j]
			};

			// 'a' must be square and symmetric.
			static SEigen EigenSymmetric(const TMatrixType& a);

			// ---- SVD ----

			// A = U * diag(s) * V^T for an m x n matrix, k = min(m, n) (one-sided Jacobi).
			struct SSVD
			{
				TMatrixType u;	// m x k, orthonormal columns
				TVectorType s;	// k singular values, non-negative, the largest first
				TMatrixType v;	// n x k, orthonormal columns
			};

			// Singular values that are zero up to rounding are returned as exactly 0.
			// When there are such values (the rank is less than k), the matching columns
			// of 'u' (of 'v' when m < n) are zero vectors, not unit vectors.
			static SSVD SVD(const TMatrixType& a);
			// Moore-Penrose pseudo-inverse, n x m. Singular values that are zero up to
			// rounding are treated as zero.
			static TMatrixType PseudoInverse(const TMatrixType& a);

		private:
			static_assert(std::is_floating_point<T>::value, "TDecompositions: T must be float or double");
		};

		typedef TDecompositions<double> CDecompositions;
		typedef TDecompositions<float> CDecompositionsF;
	}
}
