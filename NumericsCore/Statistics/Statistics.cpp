#include "Statistics.h"

#include <cmath>
#include <limits>
#include <string>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace numerics
	{
		namespace
		{
			// The elements of a matrix as nCount lines of nLength values each: the columns
			// for axisRows (reduce down the rows), the rows for axisCols.
			template<class T>
			struct TLines
			{
				TLines(const TMatrix<T>& matrix, typename TMatrix<T>::EAxis axis) : m_pData(matrix.Data())
				{
					if (axis == TMatrix<T>::axisRows)
					{
						nCount = matrix.Cols();
						nLength = matrix.Rows();
						m_nLineStride = 1;
						m_nValueStride = matrix.Cols();
					}
					else
					{
						nCount = matrix.Rows();
						nLength = matrix.Cols();
						m_nLineStride = matrix.Cols();
						m_nValueStride = 1;
					}
				}

				// Value nValue of line nLine.
				size_t Index(size_t nLine, size_t nValue) const { return nLine * m_nLineStride + nValue * m_nValueStride; }
				const T& operator ()(size_t nLine, size_t nValue) const { return m_pData[Index(nLine, nValue)]; }

				size_t nCount;
				size_t nLength;

			private:
				const T* m_pData;
				size_t m_nLineStride;
				size_t m_nValueStride;
			};

			// The divisor of a variance of n values. Throws when there are too few values.
			template<class T>
			T VarianceDivisor(size_t n, EVarianceType type, const char* pszName)
			{
				const size_t nMin = type == vtSample ? 2 : 1;
				if (n < nMin)
					throw CommonLib::CExcBase("{0}: {1} values, at least {2} are needed", std::string(pszName), n, nMin);

				return T(type == vtSample ? n - 1 : n);
			}

			// Variance of n values read through 'value(i)'. Two passes: the mean first,
			// then the squared deviations from it. The one-pass formula
			// mean(x^2) - mean(x)^2 subtracts two large close numbers and loses precision.
			template<class T, class TGet>
			T VarianceOf(size_t n, T divisor, const TGet& value)
			{
				T mean = 0;
				for (size_t i = 0; i < n; ++i)
					mean += value(i);

				mean /= T(n);

				T sum = 0;
				T residual = 0;	// sum of the deviations: 0 in exact arithmetic
				for (size_t i = 0; i < n; ++i)
				{
					const T deviation = value(i) - mean;
					sum += deviation * deviation;
					residual += deviation;
				}

				// Corrects the rounding error of the mean.
				return (sum - residual * residual / T(n)) / divisor;
			}

			template<class T, class TGet>
			T LogSumExpOf(size_t n, const TGet& value)
			{
				if (n == 0)
					return -std::numeric_limits<T>::infinity();

				T maxValue = value(0);
				for (size_t i = 1; i < n; ++i)
				{
					if (value(i) > maxValue)
						maxValue = value(i);
				}

				// All values are -infinity (the sum of exp is 0), or one is +infinity.
				if (std::isinf(maxValue))
					return maxValue;

				T sum = 0;
				for (size_t i = 0; i < n; ++i)
					sum += std::exp(value(i) - maxValue);

				return maxValue + std::log(sum);
			}

			// 'value(i)' reads the input, 'result(i)' is where the output goes.
			template<class T, class TGet, class TSet>
			void SoftmaxOf(size_t n, const TGet& value, const TSet& result)
			{
				if (n == 0)
					return;

				T maxValue = value(0);
				for (size_t i = 1; i < n; ++i)
				{
					if (value(i) > maxValue)
						maxValue = value(i);
				}

				// The largest exponent is exp(0) = 1, so the sum is at least 1.
				T sum = 0;
				for (size_t i = 0; i < n; ++i)
				{
					const T e = std::exp(value(i) - maxValue);
					result(i) = e;
					sum += e;
				}

				for (size_t i = 0; i < n; ++i)
					result(i) /= sum;
			}
		}

		// ---- Mean ----

		template<class T>
		T TStatistics<T>::Mean(const TVector<T>& vec)
		{
			return vec.Mean();
		}

		template<class T>
		TVector<T> TStatistics<T>::Mean(const TMatrix<T>& matrix, EAxis axis)
		{
			return matrix.Mean(axis);
		}

		// ---- Variance ----

		template<class T>
		T TStatistics<T>::Variance(const TVector<T>& vec, EVarianceType type)
		{
			const T divisor = VarianceDivisor<T>(vec.Size(), type, "Variance");
			const T* pData = vec.Data();

			return VarianceOf(vec.Size(), divisor, [pData](size_t i) { return pData[i]; });
		}

		template<class T>
		TVector<T> TStatistics<T>::Variance(const TMatrix<T>& matrix, EAxis axis, EVarianceType type)
		{
			const TLines<T> lines(matrix, axis);
			const T divisor = VarianceDivisor<T>(lines.nLength, type, "Variance");

			TVector<T> result(lines.nCount, matrix.GetAlloc());
			for (size_t k = 0; k < lines.nCount; ++k)
				result[k] = VarianceOf(lines.nLength, divisor, [&lines, k](size_t i) { return lines(k, i); });

			return result;
		}

		// ---- Covariance ----

		template<class T>
		TMatrix<T> TStatistics<T>::Covariance(const TMatrix<T>& matrix, EVarianceType type)
		{
			const size_t nRows = matrix.Rows();
			const size_t nCols = matrix.Cols();
			const T divisor = VarianceDivisor<T>(nRows, type, "Covariance");

			// Every column minus its mean.
			const TVector<T> means = matrix.Mean(TMatrix<T>::axisRows);
			TMatrix<T> centered(matrix);
			for (size_t i = 0; i < nRows; ++i)
			{
				T* pRow = centered.Data() + i * nCols;
				for (size_t j = 0; j < nCols; ++j)
					pRow[j] -= means[j];
			}

			// centered^T * centered / divisor. The sums go row by row, in memory order;
			// only the upper triangle is computed and then mirrored, so the result is
			// exactly symmetric.
			TMatrix<T> result(nCols, nCols, matrix.GetAlloc());
			T* pResult = result.Data();
			for (size_t i = 0; i < nRows; ++i)
			{
				const T* pRow = centered.Data() + i * nCols;
				for (size_t j = 0; j < nCols; ++j)
				{
					const T value = pRow[j];
					T* pResultRow = pResult + j * nCols;
					for (size_t k = j; k < nCols; ++k)
						pResultRow[k] += value * pRow[k];
				}
			}

			for (size_t j = 0; j < nCols; ++j)
			{
				for (size_t k = j; k < nCols; ++k)
				{
					pResult[j * nCols + k] /= divisor;
					pResult[k * nCols + j] = pResult[j * nCols + k];
				}
			}

			return result;
		}

		// ---- Log-sum-exp ----

		template<class T>
		T TStatistics<T>::LogSumExp(const TVector<T>& vec)
		{
			const T* pData = vec.Data();
			return LogSumExpOf<T>(vec.Size(), [pData](size_t i) { return pData[i]; });
		}

		template<class T>
		TVector<T> TStatistics<T>::LogSumExp(const TMatrix<T>& matrix, EAxis axis)
		{
			const TLines<T> lines(matrix, axis);

			TVector<T> result(lines.nCount, matrix.GetAlloc());
			for (size_t k = 0; k < lines.nCount; ++k)
				result[k] = LogSumExpOf<T>(lines.nLength, [&lines, k](size_t i) { return lines(k, i); });

			return result;
		}

		// ---- Softmax ----

		template<class T>
		TVector<T> TStatistics<T>::Softmax(const TVector<T>& vec)
		{
			TVector<T> result(vec.Size(), vec.GetAlloc());

			const T* pData = vec.Data();
			T* pResult = result.Data();
			SoftmaxOf<T>(vec.Size(), [pData](size_t i) { return pData[i]; }, [pResult](size_t i) -> T& { return pResult[i]; });

			return result;
		}

		template<class T>
		TMatrix<T> TStatistics<T>::Softmax(const TMatrix<T>& matrix, EAxis axis)
		{
			const TLines<T> lines(matrix, axis);

			TMatrix<T> result(matrix.Rows(), matrix.Cols(), matrix.GetAlloc());
			T* pResult = result.Data();
			for (size_t k = 0; k < lines.nCount; ++k)
			{
				SoftmaxOf<T>(lines.nLength, [&lines, k](size_t i) { return lines(k, i); },
					[&lines, pResult, k](size_t i) -> T& { return pResult[lines.Index(k, i)]; });
			}

			return result;
		}

		// The element types the library supports.
		template class TStatistics<float>;
		template class TStatistics<double>;
	}
}
