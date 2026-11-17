#include "Decompositions.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace numerics
	{
		namespace
		{
			// Jacobi methods converge in about 10 sweeps; more than this means bad input (NaN).
			const int c_nMaxSweeps = 100;

			template<class T>
			void CheckSquare(const TMatrix<T>& a, const char* pszName)
			{
				if (a.Rows() != a.Cols())
					throw CommonLib::CExcBase("{0}: the matrix must be square, size {1} x {2}", std::string(pszName), a.Rows(), a.Cols());
			}

			template<class T>
			void CheckSize(size_t nExpected, const TVector<T>& b, const char* pszName)
			{
				if (b.Size() != nExpected)
					throw CommonLib::CExcBase("{0}: the vector has size {1}, expected {2}", std::string(pszName), b.Size(), nExpected);
			}

			// Indexes 0..n-1 ordered so that values[order[0]] is the largest.
			template<class T>
			std::vector<size_t> DescendingOrder(const TVector<T>& values)
			{
				std::vector<size_t> order(values.Size());
				std::iota(order.begin(), order.end(), size_t(0));
				std::stable_sort(order.begin(), order.end(), [&values](size_t a, size_t b) { return values[a] > values[b]; });
				return order;
			}
		}

		// ---- LU ----

		template<class T>
		typename TDecompositions<T>::SLU TDecompositions<T>::LU(const TMatrix<T>& a)
		{
			CheckSquare(a, "LU");

			const size_t n = a.Rows();

			SLU result;
			result.lu = a;
			result.pivots.resize(n);

			TMatrix<T>& lu = result.lu;
			for (size_t k = 0; k < n; ++k)
			{
				// Partial pivoting: the largest element of column k, on or below the diagonal.
				size_t nPivot = k;
				TValue maxValue = std::fabs(lu(k, k));
				for (size_t i = k + 1; i < n; ++i)
				{
					if (std::fabs(lu(i, k)) > maxValue)
					{
						maxValue = std::fabs(lu(i, k));
						nPivot = i;
					}
				}

				result.pivots[k] = nPivot;

				if (maxValue == 0)
				{
					// The column is already zero below the diagonal: nothing to eliminate.
					result.singular = true;
					continue;
				}

				if (nPivot != k)
				{
					for (size_t j = 0; j < n; ++j)
						std::swap(lu(k, j), lu(nPivot, j));

					result.sign = -result.sign;
				}

				for (size_t i = k + 1; i < n; ++i)
				{
					lu(i, k) /= lu(k, k);

					const TValue factor = lu(i, k);
					for (size_t j = k + 1; j < n; ++j)
						lu(i, j) -= factor * lu(k, j);
				}
			}

			return result;
		}

		template<class T>
		TVector<T> TDecompositions<T>::SolveLU(const SLU& lu, const TVector<T>& b)
		{
			const TMatrix<T>& m = lu.lu;
			const size_t n = m.Rows();

			CheckSize(n, b, "LU solve");
			if (lu.singular)
				throw CommonLib::CExcBase("LU solve: the matrix is singular");

			TVector<T> x(b);

			// P * b: the same row swaps as in the decomposition.
			for (size_t k = 0; k < n; ++k)
			{
				if (lu.pivots[k] != k)
					std::swap(x[k], x[lu.pivots[k]]);
			}

			// L * y = P * b, forward substitution (the diagonal of L is ones).
			for (size_t i = 0; i < n; ++i)
			{
				TValue sum = x[i];
				for (size_t j = 0; j < i; ++j)
					sum -= m(i, j) * x[j];

				x[i] = sum;
			}

			// U * x = y, back substitution.
			for (size_t i = n; i-- > 0;)
			{
				TValue sum = x[i];
				for (size_t j = i + 1; j < n; ++j)
					sum -= m(i, j) * x[j];

				x[i] = sum / m(i, i);
			}

			return x;
		}

		template<class T>
		T TDecompositions<T>::Determinant(const SLU& lu)
		{
			if (lu.singular)
				return 0;

			TValue det = TValue(lu.sign);
			for (size_t i = 0; i < lu.lu.Rows(); ++i)
				det *= lu.lu(i, i);

			return det;
		}

		template<class T>
		TVector<T> TDecompositions<T>::Solve(const TMatrix<T>& a, const TVector<T>& b)
		{
			return SolveLU(LU(a), b);
		}

		template<class T>
		T TDecompositions<T>::Determinant(const TMatrix<T>& a)
		{
			return Determinant(LU(a));
		}

		// ---- Cholesky ----

		template<class T>
		TMatrix<T> TDecompositions<T>::Cholesky(const TMatrix<T>& a)
		{
			CheckSquare(a, "Cholesky");

			const size_t n = a.Rows();
			TMatrix<T> l(n, n, a.GetAlloc());

			for (size_t j = 0; j < n; ++j)
			{
				TValue diag = a(j, j);
				for (size_t k = 0; k < j; ++k)
					diag -= l(j, k) * l(j, k);

				// Written as !(> 0) so that NaN fails too.
				if (!(diag > 0))
					throw CommonLib::CExcBase("Cholesky: the matrix is not positive definite (column {0})", j);

				l(j, j) = std::sqrt(diag);

				for (size_t i = j + 1; i < n; ++i)
				{
					TValue sum = a(i, j);
					for (size_t k = 0; k < j; ++k)
						sum -= l(i, k) * l(j, k);

					l(i, j) = sum / l(j, j);
				}
			}

			return l;
		}

		template<class T>
		TVector<T> TDecompositions<T>::SolveCholesky(const TMatrix<T>& l, const TVector<T>& b)
		{
			CheckSquare(l, "Cholesky solve");

			const size_t n = l.Rows();
			CheckSize(n, b, "Cholesky solve");

			TVector<T> x(b);

			// L * y = b, forward substitution.
			for (size_t i = 0; i < n; ++i)
			{
				TValue sum = x[i];
				for (size_t j = 0; j < i; ++j)
					sum -= l(i, j) * x[j];

				if (l(i, i) == 0)
					throw CommonLib::CExcBase("Cholesky solve: zero on the diagonal (row {0})", i);

				x[i] = sum / l(i, i);
			}

			// L^T * x = y, back substitution.
			for (size_t i = n; i-- > 0;)
			{
				TValue sum = x[i];
				for (size_t j = i + 1; j < n; ++j)
					sum -= l(j, i) * x[j];

				x[i] = sum / l(i, i);
			}

			return x;
		}

		// ---- QR ----

		template<class T>
		typename TDecompositions<T>::SQR TDecompositions<T>::QR(const TMatrix<T>& a)
		{
			const size_t m = a.Rows();
			const size_t n = a.Cols();

			if (m < n)
				throw CommonLib::CExcBase("QR: the matrix must have at least as many rows as columns, size {0} x {1}", m, n);

			TMatrix<T> work(a);						// becomes R in its upper triangle
			TMatrix<T> reflectors(m, n, a.GetAlloc());	// column k: the Householder vector of step k
			std::vector<TValue> norms2(n, 0);		// its squared length; 0: no reflection at step k

			for (size_t k = 0; k < n; ++k)
			{
				TValue norm = 0;
				for (size_t i = k; i < m; ++i)
					norm += work(i, k) * work(i, k);

				norm = std::sqrt(norm);
				if (norm == 0)
					continue;	// the column is already zero on and below the diagonal

				// The reflection maps the column to (alpha, 0, ..., 0). The sign of alpha is
				// opposite to work(k, k), so the subtraction below does not cancel.
				const TValue alpha = work(k, k) > 0 ? -norm : norm;

				for (size_t i = k; i < m; ++i)
					reflectors(i, k) = work(i, k);

				reflectors(k, k) -= alpha;

				TValue norm2 = 0;
				for (size_t i = k; i < m; ++i)
					norm2 += reflectors(i, k) * reflectors(i, k);

				norms2[k] = norm2;

				// work = (I - 2 * v * v^T / (v^T * v)) * work, columns k..n-1.
				for (size_t j = k; j < n; ++j)
				{
					TValue dot = 0;
					for (size_t i = k; i < m; ++i)
						dot += reflectors(i, k) * work(i, j);

					const TValue factor = 2 * dot / norm2;
					for (size_t i = k; i < m; ++i)
						work(i, j) -= factor * reflectors(i, k);
				}
			}

			SQR result;

			result.r = TMatrix<T>(n, n, a.GetAlloc());
			for (size_t i = 0; i < n; ++i)
			{
				for (size_t j = i; j < n; ++j)
					result.r(i, j) = work(i, j);
			}

			// Q = H(0) * H(1) * ... * H(n-1) applied to the first n columns of the identity.
			result.q = TMatrix<T>(m, n, a.GetAlloc());
			for (size_t i = 0; i < n; ++i)
				result.q(i, i) = 1;

			for (size_t k = n; k-- > 0;)
			{
				if (norms2[k] == 0)
					continue;

				for (size_t j = 0; j < n; ++j)
				{
					TValue dot = 0;
					for (size_t i = k; i < m; ++i)
						dot += reflectors(i, k) * result.q(i, j);

					const TValue factor = 2 * dot / norms2[k];
					for (size_t i = k; i < m; ++i)
						result.q(i, j) -= factor * reflectors(i, k);
				}
			}

			return result;
		}

		template<class T>
		TVector<T> TDecompositions<T>::LeastSquares(const TMatrix<T>& a, const TVector<T>& b)
		{
			const size_t m = a.Rows();
			const size_t n = a.Cols();

			CheckSize(m, b, "Least squares");

			const SQR qr = QR(a);

			// |A * x - b| is minimal when R * x = Q^T * b.
			TValue maxDiag = 0;
			for (size_t i = 0; i < n; ++i)
				maxDiag = (std::max)(maxDiag, std::fabs(qr.r(i, i)));

			const TValue tolerance = std::numeric_limits<T>::epsilon() * TValue(m) * maxDiag;

			TVector<T> x(n, a.GetAlloc());
			for (size_t j = 0; j < n; ++j)
			{
				TValue sum = 0;
				for (size_t i = 0; i < m; ++i)
					sum += qr.q(i, j) * b[i];

				x[j] = sum;
			}

			for (size_t i = n; i-- > 0;)
			{
				if (!(std::fabs(qr.r(i, i)) > tolerance))
					throw CommonLib::CExcBase("Least squares: the columns of the matrix are linearly dependent (column {0})", i);

				TValue sum = x[i];
				for (size_t j = i + 1; j < n; ++j)
					sum -= qr.r(i, j) * x[j];

				x[i] = sum / qr.r(i, i);
			}

			return x;
		}

		// ---- Eigen-decomposition ----

		template<class T>
		typename TDecompositions<T>::SEigen TDecompositions<T>::EigenSymmetric(const TMatrix<T>& a)
		{
			CheckSquare(a, "Eigen");

			const size_t n = a.Rows();

			TValue maxValue = 0;
			for (size_t i = 0; i < n * n; ++i)
				maxValue = (std::max)(maxValue, std::fabs(a.Data()[i]));

			// Symmetric up to the rounding of T: about 1e-8 of the largest element for
			// double, 3e-4 for float.
			const TValue tolerance = std::sqrt(std::numeric_limits<T>::epsilon()) * maxValue;

			for (size_t i = 0; i < n; ++i)
			{
				for (size_t j = i + 1; j < n; ++j)
				{
					if (!(std::fabs(a(i, j) - a(j, i)) <= tolerance))
						throw CommonLib::CExcBase("Eigen: the matrix is not symmetric (elements ({0}, {1}) and ({1}, {0}))", i, j);
				}
			}

			TMatrix<T> work(a);					// becomes diagonal: the eigenvalues
			TMatrix<T> vectors(n, n, a.GetAlloc());	// product of the rotations: the eigenvectors
			for (size_t i = 0; i < n; ++i)
				vectors(i, i) = 1;

			// Off-diagonal elements this small do not change the eigenvalues in double
			// precision: they are set to zero instead of being rotated away.
			TValue norm = 0;
			for (size_t i = 0; i < n * n; ++i)
				norm += a.Data()[i] * a.Data()[i];

			const TValue threshold = std::numeric_limits<T>::epsilon() * std::sqrt(norm);

			bool bConverged = false;
			for (int nSweep = 0; nSweep < c_nMaxSweeps && !bConverged; ++nSweep)
			{
				// Converged when a whole sweep makes no rotation.
				bConverged = true;

				for (size_t p = 0; p < n; ++p)
				{
					for (size_t q = p + 1; q < n; ++q)
					{
						if (std::fabs(work(p, q)) <= threshold)
						{
							work(p, q) = 0;
							work(q, p) = 0;
							continue;
						}

						bConverged = false;

						// Rotation in the (p, q) plane that makes work(p, q) zero.
						const TValue theta = (work(q, q) - work(p, p)) / (2 * work(p, q));
						const TValue t = (theta >= 0 ? 1 : -1) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
						const TValue c = 1 / std::sqrt(t * t + 1);
						const TValue s = t * c;

						// work = J^T * work * J; vectors = vectors * J.
						for (size_t k = 0; k < n; ++k)
						{
							const TValue kp = work(k, p);
							const TValue kq = work(k, q);
							work(k, p) = c * kp - s * kq;
							work(k, q) = s * kp + c * kq;
						}

						for (size_t k = 0; k < n; ++k)
						{
							const TValue pk = work(p, k);
							const TValue qk = work(q, k);
							work(p, k) = c * pk - s * qk;
							work(q, k) = s * pk + c * qk;
						}

						for (size_t k = 0; k < n; ++k)
						{
							const TValue kp = vectors(k, p);
							const TValue kq = vectors(k, q);
							vectors(k, p) = c * kp - s * kq;
							vectors(k, q) = s * kp + c * kq;
						}
					}
				}
			}

			if (!bConverged)
				throw CommonLib::CExcBase("Eigen: no convergence after {0} sweeps", c_nMaxSweeps);

			TVector<T> values(n, a.GetAlloc());
			for (size_t i = 0; i < n; ++i)
				values[i] = work(i, i);

			const std::vector<size_t> order = DescendingOrder(values);

			SEigen result;
			result.values = TVector<T>(n, a.GetAlloc());
			result.vectors = TMatrix<T>(n, n, a.GetAlloc());
			for (size_t j = 0; j < n; ++j)
			{
				result.values[j] = values[order[j]];
				for (size_t i = 0; i < n; ++i)
					result.vectors(i, j) = vectors(i, order[j]);
			}

			return result;
		}

		// ---- SVD ----

		template<class T>
		typename TDecompositions<T>::SSVD TDecompositions<T>::SVD(const TMatrix<T>& a)
		{
			const size_t m = a.Rows();
			const size_t n = a.Cols();

			if (m < n)
			{
				// A^T = U * S * V^T  =>  A = V * S * U^T.
				SSVD result = SVD(TMatrix<T>::Transpose(a));
				result.u.Swap(result.v);
				return result;
			}

			// One-sided Jacobi: rotate pairs of columns of 'u' until all columns are
			// orthogonal. Then u = U * diag(s), and the product of the rotations is V.
			TMatrix<T> u(a);
			TMatrix<T> v(n, n, a.GetAlloc());
			for (size_t i = 0; i < n; ++i)
				v(i, i) = 1;

			bool bConverged = false;
			for (int nSweep = 0; nSweep < c_nMaxSweeps && !bConverged; ++nSweep)
			{
				bConverged = true;

				for (size_t p = 0; p < n; ++p)
				{
					for (size_t q = p + 1; q < n; ++q)
					{
						TValue alpha = 0;	// |column p|^2
						TValue beta = 0;	// |column q|^2
						TValue gamma = 0;	// column p . column q
						for (size_t k = 0; k < m; ++k)
						{
							alpha += u(k, p) * u(k, p);
							beta += u(k, q) * u(k, q);
							gamma += u(k, p) * u(k, q);
						}

						// Orthogonal up to rounding: nothing to do for this pair.
						if (std::fabs(gamma) <= std::numeric_limits<T>::epsilon() * std::sqrt(alpha * beta))
							continue;

						bConverged = false;

						const TValue zeta = (beta - alpha) / (2 * gamma);
						const TValue t = (zeta >= 0 ? 1 : -1) / (std::fabs(zeta) + std::sqrt(zeta * zeta + 1));
						const TValue c = 1 / std::sqrt(t * t + 1);
						const TValue s = t * c;

						for (size_t k = 0; k < m; ++k)
						{
							const TValue kp = u(k, p);
							const TValue kq = u(k, q);
							u(k, p) = c * kp - s * kq;
							u(k, q) = s * kp + c * kq;
						}

						for (size_t k = 0; k < n; ++k)
						{
							const TValue kp = v(k, p);
							const TValue kq = v(k, q);
							v(k, p) = c * kp - s * kq;
							v(k, q) = s * kp + c * kq;
						}
					}
				}
			}

			if (!bConverged)
				throw CommonLib::CExcBase("SVD: no convergence after {0} sweeps", c_nMaxSweeps);

			// The singular values are the lengths of the columns of 'u'.
			TVector<T> values(n, a.GetAlloc());
			for (size_t j = 0; j < n; ++j)
			{
				TValue norm = 0;
				for (size_t k = 0; k < m; ++k)
					norm += u(k, j) * u(k, j);

				values[j] = std::sqrt(norm);
			}

			const std::vector<size_t> order = DescendingOrder(values);
			const TValue tolerance = n == 0 ? 0 : std::numeric_limits<T>::epsilon() * TValue(m) * values[order[0]];

			SSVD result;
			result.u = TMatrix<T>(m, n, a.GetAlloc());
			result.s = TVector<T>(n, a.GetAlloc());
			result.v = TMatrix<T>(n, n, a.GetAlloc());
			for (size_t j = 0; j < n; ++j)
			{
				const size_t nSrc = order[j];
				const TValue value = values[nSrc];

				if (value > tolerance)
				{
					result.s[j] = value;
					for (size_t k = 0; k < m; ++k)
						result.u(k, j) = u(k, nSrc) / value;
				}
				// else: a zero singular value; s[j] and column j of 'u' stay zero

				for (size_t k = 0; k < n; ++k)
					result.v(k, j) = v(k, nSrc);
			}

			return result;
		}

		template<class T>
		TMatrix<T> TDecompositions<T>::PseudoInverse(const TMatrix<T>& a)
		{
			const SSVD svd = SVD(a);

			const size_t m = a.Rows();
			const size_t n = a.Cols();
			const size_t nCount = svd.s.Size();

			// A+ = V * diag(1 / s) * U^T, without the zero singular values.
			TMatrix<T> result(n, m, a.GetAlloc());
			for (size_t k = 0; k < nCount; ++k)
			{
				if (svd.s[k] == 0)
					continue;

				for (size_t i = 0; i < n; ++i)
				{
					const TValue factor = svd.v(i, k) / svd.s[k];
					for (size_t j = 0; j < m; ++j)
						result(i, j) += factor * svd.u(j, k);
				}
			}

			return result;
		}

		// The element types the library supports.
		template class TDecompositions<float>;
		template class TDecompositions<double>;
	}
}
