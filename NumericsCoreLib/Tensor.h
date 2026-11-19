#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "CommonLib/alloc/alloc.h"
#include "Matrix.h"
#include "Vector.h"

namespace ML
{
	namespace numerics
	{
		// Dense tensor: an array with any number of dimensions (axes). A vector has one
		// axis, a matrix two; a batch of color images has four (image, channel, row,
		// column). T is the element type: int, float or double (the member functions are
		// compiled for these three in Tensor.cpp).
		//
		// The shape lists the size of every axis: {2, 3, 4} is 2 blocks of 3 rows of 4
		// columns. The elements are in one contiguous buffer with the last axis changing
		// fastest (row-major, as in TMatrix): element (i, j, k) of a {d0, d1, d2} tensor
		// is at Data()[(i * d1 + j) * d2 + k].
		//
		// The buffer comes from a CommonLib allocator; when none is given, a CSimpleAlloc
		// is created. Wrong shapes, axes and indexes throw CommonLib::CExcBase.
		template<class T>
		class TTensor
		{
		public:
			typedef T TValue;
			// The type of a mean: T for float and double, double for int.
			typedef typename std::conditional<std::is_floating_point<T>::value, T, double>::type TMeanValue;
			// The size of every axis; also used for the indexes of an element.
			typedef std::vector<size_t> TShape;

			// Empty tensor: no axes, no elements.
			explicit TTensor(CommonLib::IAllocPtr pAlloc = CommonLib::IAllocPtr());
			// Tensor of the given shape filled with zeros: TTensor<double> t({2, 3, 4}).
			// A shape written in braces that starts with a literal 0 must be spelled
			// TShape({0, 3}): the compiler cannot tell {0, 3} from a null allocator.
			explicit TTensor(const TShape& shape, CommonLib::IAllocPtr pAlloc = CommonLib::IAllocPtr());
			// Tensor of the given shape filled with 'value'.
			TTensor(const TShape& shape, TValue value, CommonLib::IAllocPtr pAlloc = CommonLib::IAllocPtr());

			// A copy shares the allocator of the source and gets its own buffer.
			TTensor(const TTensor& tensor);
			// A move takes the buffer; the source becomes an empty tensor.
			TTensor(TTensor&& tensor) noexcept;
			TTensor& operator = (const TTensor& tensor);
			TTensor& operator = (TTensor&& tensor) noexcept;
			~TTensor();

			// ---- Shape ----

			// The number of axes: 1 for a vector, 2 for a matrix; 0 for an empty tensor.
			size_t Rank() const { return m_shape.size(); }
			const TShape& GetShape() const { return m_shape; }
			// The size of one axis.
			size_t Dim(size_t nAxis) const;
			// The number of elements: the product of the sizes of the axes.
			size_t Size() const { return m_nSize; }
			bool Empty() const { return m_nSize == 0; }

			// ---- Elements ----

			// Element access without a range check (assert in debug builds): one index
			// per axis, t(i, j, k).
			template<class... TIndex>
			TValue& operator ()(TIndex... indexes) { return m_pData[Offset(indexes...)]; }
			template<class... TIndex>
			const TValue& operator ()(TIndex... indexes) const { return m_pData[Offset(indexes...)]; }

			// Element access with a range check: t.At({i, j, k}).
			TValue& At(const TShape& indexes);
			const TValue& At(const TShape& indexes) const;

			TValue* Data() { return m_pData; }
			const TValue* Data() const { return m_pData; }

			void Fill(TValue value);
			// New shape; the old content is not kept, the elements are zeros.
			void Resize(const TShape& shape);
			// Another shape for the same elements, in the same order: {2, 6} -> {3, 4},
			// or {4, 3, 2} -> {4, 6} to flatten the last two axes. Nothing is copied.
			// Throws when the new shape has another number of elements.
			void Reshape(const TShape& shape);
			// Frees the buffer; the tensor becomes empty, with no axes.
			void Clear();
			void Swap(TTensor& tensor) noexcept;

			const CommonLib::IAllocPtr& GetAlloc() const { return m_pAlloc; }

			// ---- Element-wise operations ----

			// Tensors of different shapes throw. The result of +, - and * uses the
			// allocator of this tensor.
			TTensor& operator += (const TTensor& tensor);
			TTensor& operator -= (const TTensor& tensor);
			TTensor& operator *= (TValue value);		// multiply every element by a scalar
			TTensor operator + (const TTensor& tensor) const;
			TTensor operator - (const TTensor& tensor) const;
			TTensor operator * (TValue value) const;
			friend TTensor operator * (TValue value, const TTensor& tensor) { return tensor * value; }

			// ---- Reductions ----

			// Over all elements. Mean and Max of a tensor without elements throw; its Sum is 0.
			TValue Sum() const;
			TMeanValue Mean() const;
			TValue Max() const;

			// Along one axis: the axis is removed and every remaining position holds
			// the sum (mean, largest) of the elements along it. A {2, 3, 4} tensor
			// reduced along axis 1 gives {2, 4}. Reducing a tensor with one axis gives
			// a tensor of shape {1}. Mean and Max throw when the axis has size 0.
			TTensor Sum(size_t nAxis) const;
			TTensor<TMeanValue> Mean(size_t nAxis) const;
			TTensor Max(size_t nAxis) const;

			// ---- Reordering the axes ----

			// The same elements with the axes in another order: axis i of the result is
			// axis axes[i] of 'tensor'. 'axes' holds every axis number exactly once.
			// {1, 0} transposes a matrix; {0, 2, 3, 1} turns (image, channel, row,
			// column) into (image, row, column, channel).
			static TTensor Permute(const TTensor& tensor, const TShape& axes);

			// ---- Vectors and matrices ----

			// A tensor with one axis from a vector, with two axes from a matrix. The
			// result uses the allocator of the argument.
			static TTensor FromVector(const TVector<T>& vec);
			static TTensor FromMatrix(const TMatrix<T>& matrix);
			// The other way; throw when the tensor does not have one (two) axes.
			TVector<T> ToVector() const;
			TMatrix<T> ToMatrix() const;

		private:
			void Create(const TShape& shape);
			void CheckSameShape(const TTensor& tensor, const char* pszName) const;
			void CheckAxis(size_t nAxis, const char* pszName) const;
			size_t CheckedOffset(const TShape& indexes) const;

			// The position of an element in the buffer.
			template<class... TIndex>
			size_t Offset(TIndex... indexes) const
			{
				static_assert(sizeof...(indexes) > 0, "TTensor: at least one index is needed");
				assert(sizeof...(indexes) == m_shape.size());

				const size_t values[] = { size_t(indexes)... };

				size_t nOffset = 0;
				for (size_t i = 0; i < sizeof...(indexes); ++i)
				{
					assert(values[i] < m_shape[i]);
					nOffset = nOffset * m_shape[i] + values[i];
				}

				return nOffset;
			}

		private:
			CommonLib::IAllocPtr m_pAlloc;
			TValue* m_pData;
			TShape m_shape;
			size_t m_nSize;
		};

		typedef TTensor<double> CTensor;
		typedef TTensor<float> CTensorF;
		typedef TTensor<int> CTensorI;
	}
}
