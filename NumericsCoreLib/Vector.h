#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "CommonLib/alloc/alloc.h"

namespace ML
{
	namespace numerics
	{
		// Dense vector in one contiguous buffer. T is the element type: int, float or
		// double (the member functions are compiled for these three in Vector.cpp).
		// The buffer comes from a CommonLib allocator; when none is given, a CSimpleAlloc
		// is created.
		template<class T>
		class TVector
		{
		public:
			typedef T TValue;
			// The type of a mean: T for float and double, double for int.
			typedef typename std::conditional<std::is_floating_point<T>::value, T, double>::type TMeanValue;

			// Empty vector.
			explicit TVector(CommonLib::IAllocPtr pAlloc = CommonLib::IAllocPtr());
			// Vector of nSize elements filled with zeros.
			explicit TVector(size_t nSize, CommonLib::IAllocPtr pAlloc = CommonLib::IAllocPtr());
			// Vector of nSize elements filled with 'value'.
			TVector(size_t nSize, TValue value, CommonLib::IAllocPtr pAlloc = CommonLib::IAllocPtr());

			// A copy shares the allocator of the source and gets its own buffer.
			TVector(const TVector& vec);
			// A move takes the buffer; the source becomes an empty vector.
			TVector(TVector&& vec) noexcept;
			TVector& operator = (const TVector& vec);
			TVector& operator = (TVector&& vec) noexcept;
			~TVector();

			size_t Size() const { return m_nSize; }
			bool Empty() const { return m_nSize == 0; }

			// Element access without a range check (assert in debug builds).
			TValue& operator [](size_t nIndex);
			const TValue& operator [](size_t nIndex) const;

			// Element access with a range check. Throws CommonLib::CExcBase.
			TValue& At(size_t nIndex);
			const TValue& At(size_t nIndex) const;

			TValue* Data() { return m_pData; }
			const TValue* Data() const { return m_pData; }

			void Fill(TValue value);
			// New size; the old content is not kept, the elements are zeros.
			void Resize(size_t nSize);
			// Frees the buffer; the vector becomes empty.
			void Clear();
			void Swap(TVector& vec) noexcept;

			const CommonLib::IAllocPtr& GetAlloc() const { return m_pAlloc; }

			// Element-wise operations. Vectors of different sizes throw CommonLib::CExcBase.
			// The result of +, - and * uses the allocator of this vector.
			TVector& operator += (const TVector& vec);
			TVector& operator -= (const TVector& vec);
			TVector& operator *= (TValue value);		// multiply every element by a scalar
			TVector operator + (const TVector& vec) const;
			TVector operator - (const TVector& vec) const;
			TVector operator * (TValue value) const;
			friend TVector operator * (TValue value, const TVector& vec) { return vec * value; }

			// Reductions. Mean, Max and ArgMax of an empty vector throw CommonLib::CExcBase;
			// Sum of an empty vector is 0.
			TValue Sum() const;
			TMeanValue Mean() const;
			TValue Max() const;
			size_t ArgMax() const;		// index of the largest element, the first one on a tie

			// Operations. Sizes that do not match throw CommonLib::CExcBase.

			// Dot product: sum of a[i] * b[i]. The vectors must have the same size.
			static TValue Dot(const TVector& a, const TVector& b);

		private:
			void Create(size_t nSize);

		private:
			CommonLib::IAllocPtr m_pAlloc;
			TValue* m_pData;
			size_t m_nSize;
		};

		typedef TVector<double> CVector;
		typedef TVector<float> CVectorF;
		typedef TVector<int> CVectorI;
	}
}
