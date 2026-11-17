#include "Matrix.h"

#include <cassert>
#include <limits>
#include <utility>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace numerics
	{
		template<class T>
		TMatrix<T>::TMatrix(CommonLib::IAllocPtr pAlloc) : m_pAlloc(pAlloc), m_pData(nullptr), m_nRows(0), m_nCols(0)
		{
			if (m_pAlloc.get() == nullptr)
				m_pAlloc = CommonLib::IAlloc::CreateSimpleAlloc();
		}

		template<class T>
		TMatrix<T>::TMatrix(size_t nRows, size_t nCols, CommonLib::IAllocPtr pAlloc) : TMatrix<T>(nRows, nCols, TValue(0), pAlloc)
		{}

		template<class T>
		TMatrix<T>::TMatrix(size_t nRows, size_t nCols, TValue value, CommonLib::IAllocPtr pAlloc) : TMatrix<T>(pAlloc)
		{
			Create(nRows, nCols);
			Fill(value);
		}

		template<class T>
		TMatrix<T>::TMatrix(const TMatrix<T>& matrix) : TMatrix<T>(matrix.m_pAlloc)
		{
			Create(matrix.m_nRows, matrix.m_nCols);
			if (!Empty())
				memcpy(m_pData, matrix.m_pData, Size() * sizeof(TValue));
		}

		template<class T>
		TMatrix<T>::TMatrix(TMatrix<T>&& matrix) noexcept : m_pAlloc(matrix.m_pAlloc), m_pData(matrix.m_pData),
			m_nRows(matrix.m_nRows), m_nCols(matrix.m_nCols)
		{
			// The source keeps the allocator, so it stays usable as an empty matrix.
			matrix.m_pData = nullptr;
			matrix.m_nRows = 0;
			matrix.m_nCols = 0;
		}

		template<class T>
		TMatrix<T>& TMatrix<T>::operator = (const TMatrix<T>& matrix)
		{
			if (this != &matrix)
			{
				TMatrix<T> tmp(matrix);
				Swap(tmp);
			}

			return *this;
		}

		template<class T>
		TMatrix<T>& TMatrix<T>::operator = (TMatrix<T>&& matrix) noexcept
		{
			if (this != &matrix)
			{
				TMatrix<T> tmp(std::move(matrix));
				Swap(tmp);
			}

			return *this;
		}

		template<class T>
		TMatrix<T>::~TMatrix()
		{
			if (m_pData != nullptr)
				m_pAlloc->Free(m_pData);
		}

		template<class T>
		T& TMatrix<T>::operator ()(size_t nRow, size_t nCol)
		{
			assert(nRow < m_nRows && nCol < m_nCols);
			return m_pData[nRow * m_nCols + nCol];
		}

		template<class T>
		const T& TMatrix<T>::operator ()(size_t nRow, size_t nCol) const
		{
			assert(nRow < m_nRows && nCol < m_nCols);
			return m_pData[nRow * m_nCols + nCol];
		}

		template<class T>
		T& TMatrix<T>::At(size_t nRow, size_t nCol)
		{
			if (nRow >= m_nRows || nCol >= m_nCols)
				throw CommonLib::CExcBase("Matrix: element ({0}, {1}) is out of range, size {2} x {3}", nRow, nCol, m_nRows, m_nCols);

			return m_pData[nRow * m_nCols + nCol];
		}

		template<class T>
		const T& TMatrix<T>::At(size_t nRow, size_t nCol) const
		{
			if (nRow >= m_nRows || nCol >= m_nCols)
				throw CommonLib::CExcBase("Matrix: element ({0}, {1}) is out of range, size {2} x {3}", nRow, nCol, m_nRows, m_nCols);

			return m_pData[nRow * m_nCols + nCol];
		}

		template<class T>
		void TMatrix<T>::Fill(TValue value)
		{
			const size_t nSize = Size();
			for (size_t i = 0; i < nSize; ++i)
				m_pData[i] = value;
		}

		template<class T>
		void TMatrix<T>::Resize(size_t nRows, size_t nCols)
		{
			// Built aside first: if the allocation throws, this matrix is unchanged.
			TMatrix<T> tmp(nRows, nCols, m_pAlloc);
			Swap(tmp);
		}

		template<class T>
		void TMatrix<T>::Clear()
		{
			if (m_pData != nullptr)
			{
				m_pAlloc->Free(m_pData);
				m_pData = nullptr;
			}

			m_nRows = 0;
			m_nCols = 0;
		}

		template<class T>
		void TMatrix<T>::Swap(TMatrix<T>& matrix) noexcept
		{
			std::swap(m_pAlloc, matrix.m_pAlloc);
			std::swap(m_pData, matrix.m_pData);
			std::swap(m_nRows, matrix.m_nRows);
			std::swap(m_nCols, matrix.m_nCols);
		}

		template<class T>
		TMatrix<T>& TMatrix<T>::operator += (const TMatrix<T>& matrix)
		{
			if (m_nRows != matrix.m_nRows || m_nCols != matrix.m_nCols)
				throw CommonLib::CExcBase("Matrix add: sizes {0} x {1} and {2} x {3} do not match", m_nRows, m_nCols, matrix.m_nRows, matrix.m_nCols);

			const size_t nSize = Size();
			for (size_t i = 0; i < nSize; ++i)
				m_pData[i] += matrix.m_pData[i];

			return *this;
		}

		template<class T>
		TMatrix<T>& TMatrix<T>::operator -= (const TMatrix<T>& matrix)
		{
			if (m_nRows != matrix.m_nRows || m_nCols != matrix.m_nCols)
				throw CommonLib::CExcBase("Matrix subtract: sizes {0} x {1} and {2} x {3} do not match", m_nRows, m_nCols, matrix.m_nRows, matrix.m_nCols);

			const size_t nSize = Size();
			for (size_t i = 0; i < nSize; ++i)
				m_pData[i] -= matrix.m_pData[i];

			return *this;
		}

		template<class T>
		TMatrix<T>& TMatrix<T>::operator *= (TValue value)
		{
			const size_t nSize = Size();
			for (size_t i = 0; i < nSize; ++i)
				m_pData[i] *= value;

			return *this;
		}

		template<class T>
		TMatrix<T> TMatrix<T>::operator + (const TMatrix<T>& matrix) const
		{
			TMatrix<T> result(*this);
			result += matrix;
			return result;
		}

		template<class T>
		TMatrix<T> TMatrix<T>::operator - (const TMatrix<T>& matrix) const
		{
			TMatrix<T> result(*this);
			result -= matrix;
			return result;
		}

		template<class T>
		TMatrix<T> TMatrix<T>::operator * (TValue value) const
		{
			TMatrix<T> result(*this);
			result *= value;
			return result;
		}

		template<class T>
		T TMatrix<T>::Sum() const
		{
			const size_t nSize = Size();
			TValue sum = 0;
			for (size_t i = 0; i < nSize; ++i)
				sum += m_pData[i];

			return sum;
		}

		template<class T>
		typename TMatrix<T>::TMeanValue TMatrix<T>::Mean() const
		{
			if (Empty())
				throw CommonLib::CExcBase("Matrix mean: the matrix is empty, size {0} x {1}", m_nRows, m_nCols);

			return TMeanValue(Sum()) / TMeanValue(Size());
		}

		template<class T>
		T TMatrix<T>::Max() const
		{
			if (Empty())
				throw CommonLib::CExcBase("Matrix max: the matrix is empty, size {0} x {1}", m_nRows, m_nCols);

			const size_t nSize = Size();
			TValue value = m_pData[0];
			for (size_t i = 1; i < nSize; ++i)
			{
				if (m_pData[i] > value)
					value = m_pData[i];
			}

			return value;
		}

		template<class T>
		TVector<T> TMatrix<T>::Sum(EAxis axis) const
		{
			TVector<T> result(axis == axisRows ? m_nCols : m_nRows, m_pAlloc);
			TValue* pResult = result.Data();

			for (size_t i = 0; i < m_nRows; ++i)
			{
				const TValue* pRow = m_pData + i * m_nCols;
				if (axis == axisRows)
				{
					for (size_t j = 0; j < m_nCols; ++j)
						pResult[j] += pRow[j];
				}
				else
				{
					TValue sum = 0;
					for (size_t j = 0; j < m_nCols; ++j)
						sum += pRow[j];

					pResult[i] = sum;
				}
			}

			return result;
		}

		template<class T>
		TVector<typename TMatrix<T>::TMeanValue> TMatrix<T>::Mean(EAxis axis) const
		{
			CheckReduce(axis, "mean");

			const TVector<T> sums = Sum(axis);
			const TMeanValue count = TMeanValue(axis == axisRows ? m_nRows : m_nCols);

			TVector<TMeanValue> result(sums.Size(), m_pAlloc);
			for (size_t i = 0; i < sums.Size(); ++i)
				result[i] = TMeanValue(sums[i]) / count;

			return result;
		}

		template<class T>
		TVector<T> TMatrix<T>::Max(EAxis axis) const
		{
			const std::vector<size_t> indexes = ArgMax(axis);

			TVector<T> result(indexes.size(), m_pAlloc);
			TValue* pResult = result.Data();
			for (size_t i = 0; i < indexes.size(); ++i)
				pResult[i] = axis == axisRows ? m_pData[indexes[i] * m_nCols + i] : m_pData[i * m_nCols + indexes[i]];

			return result;
		}

		template<class T>
		std::vector<size_t> TMatrix<T>::ArgMax(EAxis axis) const
		{
			CheckReduce(axis, "max");

			std::vector<size_t> indexes(axis == axisRows ? m_nCols : m_nRows, 0);

			if (axis == axisRows)
			{
				// Row by row, so the matrix is read in memory order.
				for (size_t i = 1; i < m_nRows; ++i)
				{
					const TValue* pRow = m_pData + i * m_nCols;
					for (size_t j = 0; j < m_nCols; ++j)
					{
						if (pRow[j] > m_pData[indexes[j] * m_nCols + j])
							indexes[j] = i;
					}
				}
			}
			else
			{
				for (size_t i = 0; i < m_nRows; ++i)
				{
					const TValue* pRow = m_pData + i * m_nCols;
					size_t nMax = 0;
					for (size_t j = 1; j < m_nCols; ++j)
					{
						if (pRow[j] > pRow[nMax])
							nMax = j;
					}

					indexes[i] = nMax;
				}
			}

			return indexes;
		}

		// Mean, max and argmax need at least one element along the axis.
		template<class T>
		void TMatrix<T>::CheckReduce(EAxis axis, const char* pszName) const
		{
			if ((axis == axisRows ? m_nRows : m_nCols) == 0)
			{
				throw CommonLib::CExcBase("Matrix {0}: nothing to reduce along axis {1}, size {2} x {3}",
					std::string(pszName), int(axis), m_nRows, m_nCols);
			}
		}

		template<class T>
		TMatrix<T> TMatrix<T>::Multiply(const TMatrix<T>& a, const TMatrix<T>& b)
		{
			if (a.m_nCols != b.m_nRows)
				throw CommonLib::CExcBase("Matrix multiply: sizes {0} x {1} and {2} x {3} do not match", a.m_nRows, a.m_nCols, b.m_nRows, b.m_nCols);

			TMatrix<T> result(a.m_nRows, b.m_nCols, a.m_pAlloc);

			// Loop order i, k, j: the inner loop walks a row of 'b' and a row of 'result',
			// both contiguous in memory, instead of a column of 'b'.
			for (size_t i = 0; i < a.m_nRows; ++i)
			{
				TValue* pResultRow = result.m_pData + i * result.m_nCols;
				for (size_t k = 0; k < a.m_nCols; ++k)
				{
					const TValue value = a.m_pData[i * a.m_nCols + k];
					const TValue* pRow = b.m_pData + k * b.m_nCols;
					for (size_t j = 0; j < b.m_nCols; ++j)
						pResultRow[j] += value * pRow[j];
				}
			}

			return result;
		}

		template<class T>
		TVector<T> TMatrix<T>::Multiply(const TMatrix<T>& a, const TVector<T>& vec)
		{
			if (a.m_nCols != vec.Size())
				throw CommonLib::CExcBase("Matrix multiply: matrix {0} x {1} and vector of size {2} do not match", a.m_nRows, a.m_nCols, vec.Size());

			TVector<T> result(a.m_nRows, a.m_pAlloc);

			const TValue* pVec = vec.Data();
			TValue* pResult = result.Data();
			for (size_t i = 0; i < a.m_nRows; ++i)
			{
				const TValue* pRow = a.m_pData + i * a.m_nCols;
				TValue sum = 0;
				for (size_t j = 0; j < a.m_nCols; ++j)
					sum += pRow[j] * pVec[j];

				pResult[i] = sum;
			}

			return result;
		}

		template<class T>
		TMatrix<T> TMatrix<T>::Transpose(const TMatrix<T>& a)
		{
			TMatrix<T> result(a.m_nCols, a.m_nRows, a.m_pAlloc);

			for (size_t i = 0; i < a.m_nRows; ++i)
			{
				for (size_t j = 0; j < a.m_nCols; ++j)
					result.m_pData[j * a.m_nRows + i] = a.m_pData[i * a.m_nCols + j];
			}

			return result;
		}

		// Allocates the buffer of an empty matrix; the elements are not initialized.
		template<class T>
		void TMatrix<T>::Create(size_t nRows, size_t nCols)
		{
			assert(m_pData == nullptr);

			// A matrix with a zero dimension has no buffer (IAlloc::Alloc rejects size 0).
			if (nRows != 0 && nCols != 0)
			{
				if (nCols > (std::numeric_limits<size_t>::max)() / sizeof(TValue) / nRows)
					throw CommonLib::CExcBase("Matrix: size {0} x {1} is too big", nRows, nCols);

				m_pData = (TValue*)m_pAlloc->Alloc(nRows * nCols * sizeof(TValue));
			}

			m_nRows = nRows;
			m_nCols = nCols;
		}

		// The element types the library supports.
		template class TMatrix<int>;
		template class TMatrix<float>;
		template class TMatrix<double>;
	}
}
