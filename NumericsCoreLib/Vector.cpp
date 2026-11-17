#include "Vector.h"

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
		TVector<T>::TVector(CommonLib::IAllocPtr pAlloc) : m_pAlloc(pAlloc), m_pData(nullptr), m_nSize(0)
		{
			if (m_pAlloc.get() == nullptr)
				m_pAlloc = CommonLib::IAlloc::CreateSimpleAlloc();
		}

		template<class T>
		TVector<T>::TVector(size_t nSize, CommonLib::IAllocPtr pAlloc) : TVector<T>(nSize, TValue(0), pAlloc)
		{}

		template<class T>
		TVector<T>::TVector(size_t nSize, TValue value, CommonLib::IAllocPtr pAlloc) : TVector<T>(pAlloc)
		{
			Create(nSize);
			Fill(value);
		}

		template<class T>
		TVector<T>::TVector(const TVector<T>& vec) : TVector<T>(vec.m_pAlloc)
		{
			Create(vec.m_nSize);
			if (!Empty())
				memcpy(m_pData, vec.m_pData, m_nSize * sizeof(TValue));
		}

		template<class T>
		TVector<T>::TVector(TVector<T>&& vec) noexcept : m_pAlloc(vec.m_pAlloc), m_pData(vec.m_pData), m_nSize(vec.m_nSize)
		{
			// The source keeps the allocator, so it stays usable as an empty vector.
			vec.m_pData = nullptr;
			vec.m_nSize = 0;
		}

		template<class T>
		TVector<T>& TVector<T>::operator = (const TVector<T>& vec)
		{
			if (this != &vec)
			{
				TVector<T> tmp(vec);
				Swap(tmp);
			}

			return *this;
		}

		template<class T>
		TVector<T>& TVector<T>::operator = (TVector<T>&& vec) noexcept
		{
			if (this != &vec)
			{
				TVector<T> tmp(std::move(vec));
				Swap(tmp);
			}

			return *this;
		}

		template<class T>
		TVector<T>::~TVector()
		{
			if (m_pData != nullptr)
				m_pAlloc->Free(m_pData);
		}

		template<class T>
		T& TVector<T>::operator [](size_t nIndex)
		{
			assert(nIndex < m_nSize);
			return m_pData[nIndex];
		}

		template<class T>
		const T& TVector<T>::operator [](size_t nIndex) const
		{
			assert(nIndex < m_nSize);
			return m_pData[nIndex];
		}

		template<class T>
		T& TVector<T>::At(size_t nIndex)
		{
			if (nIndex >= m_nSize)
				throw CommonLib::CExcBase("Vector: element {0} is out of range, size {1}", nIndex, m_nSize);

			return m_pData[nIndex];
		}

		template<class T>
		const T& TVector<T>::At(size_t nIndex) const
		{
			if (nIndex >= m_nSize)
				throw CommonLib::CExcBase("Vector: element {0} is out of range, size {1}", nIndex, m_nSize);

			return m_pData[nIndex];
		}

		template<class T>
		void TVector<T>::Fill(TValue value)
		{
			for (size_t i = 0; i < m_nSize; ++i)
				m_pData[i] = value;
		}

		template<class T>
		void TVector<T>::Resize(size_t nSize)
		{
			// Built aside first: if the allocation throws, this vector is unchanged.
			TVector<T> tmp(nSize, m_pAlloc);
			Swap(tmp);
		}

		template<class T>
		void TVector<T>::Clear()
		{
			if (m_pData != nullptr)
			{
				m_pAlloc->Free(m_pData);
				m_pData = nullptr;
			}

			m_nSize = 0;
		}

		template<class T>
		void TVector<T>::Swap(TVector<T>& vec) noexcept
		{
			std::swap(m_pAlloc, vec.m_pAlloc);
			std::swap(m_pData, vec.m_pData);
			std::swap(m_nSize, vec.m_nSize);
		}

		template<class T>
		TVector<T>& TVector<T>::operator += (const TVector<T>& vec)
		{
			if (m_nSize != vec.m_nSize)
				throw CommonLib::CExcBase("Vector add: sizes {0} and {1} do not match", m_nSize, vec.m_nSize);

			const size_t nSize = m_nSize;
			for (size_t i = 0; i < nSize; ++i)
				m_pData[i] += vec.m_pData[i];

			return *this;
		}

		template<class T>
		TVector<T>& TVector<T>::operator -= (const TVector<T>& vec)
		{
			if (m_nSize != vec.m_nSize)
				throw CommonLib::CExcBase("Vector subtract: sizes {0} and {1} do not match", m_nSize, vec.m_nSize);

			const size_t nSize = m_nSize;
			for (size_t i = 0; i < nSize; ++i)
				m_pData[i] -= vec.m_pData[i];

			return *this;
		}

		template<class T>
		TVector<T>& TVector<T>::operator *= (TValue value)
		{
			const size_t nSize = m_nSize;
			for (size_t i = 0; i < nSize; ++i)
				m_pData[i] *= value;

			return *this;
		}

		template<class T>
		TVector<T> TVector<T>::operator + (const TVector<T>& vec) const
		{
			TVector<T> result(*this);
			result += vec;
			return result;
		}

		template<class T>
		TVector<T> TVector<T>::operator - (const TVector<T>& vec) const
		{
			TVector<T> result(*this);
			result -= vec;
			return result;
		}

		template<class T>
		TVector<T> TVector<T>::operator * (TValue value) const
		{
			TVector<T> result(*this);
			result *= value;
			return result;
		}

		template<class T>
		T TVector<T>::Sum() const
		{
			TValue sum = 0;
			for (size_t i = 0; i < m_nSize; ++i)
				sum += m_pData[i];

			return sum;
		}

		template<class T>
		typename TVector<T>::TMeanValue TVector<T>::Mean() const
		{
			if (m_nSize == 0)
				throw CommonLib::CExcBase("Vector mean: the vector is empty");

			return TMeanValue(Sum()) / TMeanValue(m_nSize);
		}

		template<class T>
		T TVector<T>::Max() const
		{
			return m_pData[ArgMax()];
		}

		template<class T>
		size_t TVector<T>::ArgMax() const
		{
			if (m_nSize == 0)
				throw CommonLib::CExcBase("Vector max: the vector is empty");

			size_t nMax = 0;
			for (size_t i = 1; i < m_nSize; ++i)
			{
				if (m_pData[i] > m_pData[nMax])
					nMax = i;
			}

			return nMax;
		}

		template<class T>
		T TVector<T>::Dot(const TVector<T>& a, const TVector<T>& b)
		{
			if (a.m_nSize != b.m_nSize)
				throw CommonLib::CExcBase("Vector dot: sizes {0} and {1} do not match", a.m_nSize, b.m_nSize);

			TValue sum = 0;
			for (size_t i = 0; i < a.m_nSize; ++i)
				sum += a.m_pData[i] * b.m_pData[i];

			return sum;
		}

		// Allocates the buffer of an empty vector; the elements are not initialized.
		template<class T>
		void TVector<T>::Create(size_t nSize)
		{
			assert(m_pData == nullptr);

			// An empty vector has no buffer (IAlloc::Alloc rejects size 0).
			if (nSize != 0)
			{
				if (nSize > (std::numeric_limits<size_t>::max)() / sizeof(TValue))
					throw CommonLib::CExcBase("Vector: size {0} is too big", nSize);

				m_pData = (TValue*)m_pAlloc->Alloc(nSize * sizeof(TValue));
			}

			m_nSize = nSize;
		}

		// The element types the library supports.
		template class TVector<int>;
		template class TVector<float>;
		template class TVector<double>;
	}
}
