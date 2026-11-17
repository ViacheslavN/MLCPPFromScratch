#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "CommonLib/alloc/alloc.h"
#include "Vector.h"

namespace ML
{
	namespace numerics
	{
		// Dense matrix. T is the element type: int, float or double (the member functions
		// are compiled for these three in Matrix.cpp). One contiguous buffer in row-major
		// order: element (row, col) is at Data()[row * Cols() + col]. The buffer comes
		// from a CommonLib allocator; when none is given, a CSimpleAlloc is created.
		template<class T>
		class TMatrix
		{
		public:
			typedef T TValue;
			// The type of a mean: T for float and double, double for int.
			typedef typename std::conditional<std::is_floating_point<T>::value, T, double>::type TMeanValue;

			// The dimension a reduction runs along, numbered as in NumPy.
			enum EAxis
			{
				axisRows = 0,	// down the rows: one result per column, Cols() results
				axisCols = 1	// across the columns: one result per row, Rows() results
			};

			// Empty matrix (0 x 0).
			explicit TMatrix(CommonLib::IAllocPtr pAlloc = CommonLib::IAllocPtr());
			// nRows x nCols matrix filled with zeros.
			TMatrix(size_t nRows, size_t nCols, CommonLib::IAllocPtr pAlloc = CommonLib::IAllocPtr());
			// nRows x nCols matrix filled with 'value'.
			TMatrix(size_t nRows, size_t nCols, TValue value, CommonLib::IAllocPtr pAlloc = CommonLib::IAllocPtr());

			// A copy shares the allocator of the source and gets its own buffer.
			TMatrix(const TMatrix& matrix);
			// A move takes the buffer; the source becomes an empty matrix.
			TMatrix(TMatrix&& matrix) noexcept;
			TMatrix& operator = (const TMatrix& matrix);
			TMatrix& operator = (TMatrix&& matrix) noexcept;
			~TMatrix();

			size_t Rows() const { return m_nRows; }
			size_t Cols() const { return m_nCols; }
			size_t Size() const { return m_nRows * m_nCols; }	// number of elements
			bool Empty() const { return Size() == 0; }

			// Element access without a range check (assert in debug builds).
			TValue& operator ()(size_t nRow, size_t nCol);
			const TValue& operator ()(size_t nRow, size_t nCol) const;

			// Element access with a range check. Throws CommonLib::CExcBase.
			TValue& At(size_t nRow, size_t nCol);
			const TValue& At(size_t nRow, size_t nCol) const;

			TValue* Data() { return m_pData; }
			const TValue* Data() const { return m_pData; }

			void Fill(TValue value);
			// New size; the old content is not kept, the elements are zeros.
			void Resize(size_t nRows, size_t nCols);
			// Frees the buffer; the matrix becomes 0 x 0.
			void Clear();
			void Swap(TMatrix& matrix) noexcept;

			const CommonLib::IAllocPtr& GetAlloc() const { return m_pAlloc; }

			// Element-wise operations. Matrices of different sizes throw CommonLib::CExcBase.
			// The result of +, - and * uses the allocator of this matrix.
			TMatrix& operator += (const TMatrix& matrix);
			TMatrix& operator -= (const TMatrix& matrix);
			TMatrix& operator *= (TValue value);		// multiply every element by a scalar
			TMatrix operator + (const TMatrix& matrix) const;
			TMatrix operator - (const TMatrix& matrix) const;
			TMatrix operator * (TValue value) const;
			friend TMatrix operator * (TValue value, const TMatrix& matrix) { return matrix * value; }

			// Reductions over all elements. Mean and Max of an empty matrix throw
			// CommonLib::CExcBase; Sum of an empty matrix is 0.
			TValue Sum() const;
			TMeanValue Mean() const;
			TValue Max() const;

			// Reductions along an axis. For a 2 x 3 matrix, axisRows gives 3 results (one
			// per column) and axisCols gives 2 (one per row). The result uses the allocator
			// of this matrix. Mean, Max and ArgMax throw CommonLib::CExcBase when there is
			// nothing to reduce (no rows for axisRows, no columns for axisCols).
			TVector<T> Sum(EAxis axis) const;
			TVector<TMeanValue> Mean(EAxis axis) const;
			TVector<T> Max(EAxis axis) const;
			// Index of the largest element along the axis (a row index for axisRows, a
			// column index for axisCols), the first one on a tie.
			std::vector<size_t> ArgMax(EAxis axis) const;

			// Operations. The result uses the allocator of the first argument.
			// Sizes that do not match throw CommonLib::CExcBase.

			// Matrix product a * b: (n x k) * (k x m) -> (n x m).
			static TMatrix Multiply(const TMatrix& a, const TMatrix& b);
			// Matrix-vector product a * vec: (n x m) * (m) -> (n).
			static TVector<T> Multiply(const TMatrix& a, const TVector<T>& vec);
			// Transposed matrix: (n x m) -> (m x n).
			static TMatrix Transpose(const TMatrix& a);

		private:
			void Create(size_t nRows, size_t nCols);
			void CheckReduce(EAxis axis, const char* pszName) const;

		private:
			CommonLib::IAllocPtr m_pAlloc;
			TValue* m_pData;
			size_t m_nRows;
			size_t m_nCols;
		};

		typedef TMatrix<double> CMatrix;
		typedef TMatrix<float> CMatrixF;
		typedef TMatrix<int> CMatrixI;
	}
}
