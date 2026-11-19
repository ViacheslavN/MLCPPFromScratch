#include "Tensor.h"

#include <limits>
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
			// The numbers joined by the separator, for the error messages.
			std::string Join(const std::vector<size_t>& values, const char* pszSeparator)
			{
				std::string text;
				for (size_t i = 0; i < values.size(); ++i)
					text += (i == 0 ? "" : pszSeparator) + std::to_string(values[i]);

				return text;
			}

			// "2 x 3 x 4"
			std::string ShapeText(const std::vector<size_t>& shape)
			{
				return shape.empty() ? std::string("(no axes)") : Join(shape, " x ");
			}

			// The number of elements of a shape; 0 for no axes. Throws when it does not
			// fit: nBytes is the size of one element.
			size_t ElementCount(const std::vector<size_t>& shape, size_t nBytes)
			{
				if (shape.empty())
					return 0;

				size_t nCount = 1;
				for (size_t nDim : shape)
				{
					if (nDim != 0 && nCount > (std::numeric_limits<size_t>::max)() / nBytes / nDim)
						throw CommonLib::CExcBase("Tensor: size {0} is too big", ShapeText(shape));

					nCount *= nDim;
				}

				return nCount;
			}

			// A tensor seen along one axis: nOuter blocks of nLength steps of nInner
			// elements. The element at (outer, step, inner) is at
			// (outer * nLength + step) * nInner + inner.
			struct SAxis
			{
				SAxis(const std::vector<size_t>& shape, size_t nAxis) : nOuter(1), nLength(shape[nAxis]), nInner(1)
				{
					for (size_t i = 0; i < nAxis; ++i)
						nOuter *= shape[i];

					for (size_t i = nAxis + 1; i < shape.size(); ++i)
						nInner *= shape[i];

					// The shape without the axis; a single axis leaves {1}.
					for (size_t i = 0; i < shape.size(); ++i)
					{
						if (i != nAxis)
							reduced.push_back(shape[i]);
					}

					if (reduced.empty())
						reduced.push_back(1);
				}

				size_t nOuter;
				size_t nLength;
				size_t nInner;
				std::vector<size_t> reduced;
			};
		}

		// ---- Construction ----

		template<class T>
		TTensor<T>::TTensor(CommonLib::IAllocPtr pAlloc) : m_pAlloc(pAlloc), m_pData(nullptr), m_nSize(0)
		{
			if (m_pAlloc.get() == nullptr)
				m_pAlloc = CommonLib::IAlloc::CreateSimpleAlloc();
		}

		template<class T>
		TTensor<T>::TTensor(const TShape& shape, CommonLib::IAllocPtr pAlloc) : TTensor<T>(shape, TValue(0), pAlloc)
		{}

		template<class T>
		TTensor<T>::TTensor(const TShape& shape, TValue value, CommonLib::IAllocPtr pAlloc) : TTensor<T>(pAlloc)
		{
			Create(shape);
			Fill(value);
		}

		template<class T>
		TTensor<T>::TTensor(const TTensor<T>& tensor) : TTensor<T>(tensor.m_pAlloc)
		{
			Create(tensor.m_shape);
			if (!Empty())
				memcpy(m_pData, tensor.m_pData, m_nSize * sizeof(TValue));
		}

		template<class T>
		TTensor<T>::TTensor(TTensor<T>&& tensor) noexcept : m_pAlloc(tensor.m_pAlloc), m_pData(tensor.m_pData),
			m_shape(std::move(tensor.m_shape)), m_nSize(tensor.m_nSize)
		{
			// The source keeps the allocator, so it stays usable as an empty tensor.
			tensor.m_pData = nullptr;
			tensor.m_shape.clear();
			tensor.m_nSize = 0;
		}

		template<class T>
		TTensor<T>& TTensor<T>::operator = (const TTensor<T>& tensor)
		{
			if (this != &tensor)
			{
				TTensor<T> tmp(tensor);
				Swap(tmp);
			}

			return *this;
		}

		template<class T>
		TTensor<T>& TTensor<T>::operator = (TTensor<T>&& tensor) noexcept
		{
			if (this != &tensor)
			{
				TTensor<T> tmp(std::move(tensor));
				Swap(tmp);
			}

			return *this;
		}

		template<class T>
		TTensor<T>::~TTensor()
		{
			if (m_pData != nullptr)
				m_pAlloc->Free(m_pData);
		}

		// ---- Shape and elements ----

		template<class T>
		size_t TTensor<T>::Dim(size_t nAxis) const
		{
			CheckAxis(nAxis, "Tensor");
			return m_shape[nAxis];
		}

		template<class T>
		T& TTensor<T>::At(const TShape& indexes)
		{
			return m_pData[CheckedOffset(indexes)];
		}

		template<class T>
		const T& TTensor<T>::At(const TShape& indexes) const
		{
			return m_pData[CheckedOffset(indexes)];
		}

		template<class T>
		void TTensor<T>::Fill(TValue value)
		{
			for (size_t i = 0; i < m_nSize; ++i)
				m_pData[i] = value;
		}

		template<class T>
		void TTensor<T>::Resize(const TShape& shape)
		{
			// Built aside first: if the allocation throws, this tensor is unchanged.
			TTensor<T> tmp(shape, m_pAlloc);
			Swap(tmp);
		}

		template<class T>
		void TTensor<T>::Reshape(const TShape& shape)
		{
			if (ElementCount(shape, sizeof(TValue)) != m_nSize)
			{
				throw CommonLib::CExcBase("Tensor reshape: {0} has {1} elements, {2} has another number",
					ShapeText(m_shape), m_nSize, ShapeText(shape));
			}

			m_shape = shape;
		}

		template<class T>
		void TTensor<T>::Clear()
		{
			if (m_pData != nullptr)
			{
				m_pAlloc->Free(m_pData);
				m_pData = nullptr;
			}

			m_shape.clear();
			m_nSize = 0;
		}

		template<class T>
		void TTensor<T>::Swap(TTensor<T>& tensor) noexcept
		{
			std::swap(m_pAlloc, tensor.m_pAlloc);
			std::swap(m_pData, tensor.m_pData);
			std::swap(m_shape, tensor.m_shape);
			std::swap(m_nSize, tensor.m_nSize);
		}

		// ---- Element-wise operations ----

		template<class T>
		TTensor<T>& TTensor<T>::operator += (const TTensor<T>& tensor)
		{
			CheckSameShape(tensor, "Tensor add");

			for (size_t i = 0; i < m_nSize; ++i)
				m_pData[i] += tensor.m_pData[i];

			return *this;
		}

		template<class T>
		TTensor<T>& TTensor<T>::operator -= (const TTensor<T>& tensor)
		{
			CheckSameShape(tensor, "Tensor subtract");

			for (size_t i = 0; i < m_nSize; ++i)
				m_pData[i] -= tensor.m_pData[i];

			return *this;
		}

		template<class T>
		TTensor<T>& TTensor<T>::operator *= (TValue value)
		{
			for (size_t i = 0; i < m_nSize; ++i)
				m_pData[i] *= value;

			return *this;
		}

		template<class T>
		TTensor<T> TTensor<T>::operator + (const TTensor<T>& tensor) const
		{
			TTensor<T> result(*this);
			result += tensor;
			return result;
		}

		template<class T>
		TTensor<T> TTensor<T>::operator - (const TTensor<T>& tensor) const
		{
			TTensor<T> result(*this);
			result -= tensor;
			return result;
		}

		template<class T>
		TTensor<T> TTensor<T>::operator * (TValue value) const
		{
			TTensor<T> result(*this);
			result *= value;
			return result;
		}

		// ---- Reductions ----

		template<class T>
		T TTensor<T>::Sum() const
		{
			TValue sum = 0;
			for (size_t i = 0; i < m_nSize; ++i)
				sum += m_pData[i];

			return sum;
		}

		template<class T>
		typename TTensor<T>::TMeanValue TTensor<T>::Mean() const
		{
			if (Empty())
				throw CommonLib::CExcBase("Tensor mean: the tensor has no elements, size {0}", ShapeText(m_shape));

			return TMeanValue(Sum()) / TMeanValue(m_nSize);
		}

		template<class T>
		T TTensor<T>::Max() const
		{
			if (Empty())
				throw CommonLib::CExcBase("Tensor max: the tensor has no elements, size {0}", ShapeText(m_shape));

			TValue value = m_pData[0];
			for (size_t i = 1; i < m_nSize; ++i)
			{
				if (m_pData[i] > value)
					value = m_pData[i];
			}

			return value;
		}

		template<class T>
		TTensor<T> TTensor<T>::Sum(size_t nAxis) const
		{
			CheckAxis(nAxis, "Tensor sum");

			const SAxis axis(m_shape, nAxis);

			TTensor<T> result(axis.reduced, m_pAlloc);
			for (size_t nOuter = 0; nOuter < axis.nOuter; ++nOuter)
			{
				TValue* pResult = result.m_pData + nOuter * axis.nInner;
				for (size_t nStep = 0; nStep < axis.nLength; ++nStep)
				{
					const TValue* pSource = m_pData + (nOuter * axis.nLength + nStep) * axis.nInner;
					for (size_t nInner = 0; nInner < axis.nInner; ++nInner)
						pResult[nInner] += pSource[nInner];
				}
			}

			return result;
		}

		template<class T>
		TTensor<typename TTensor<T>::TMeanValue> TTensor<T>::Mean(size_t nAxis) const
		{
			CheckAxis(nAxis, "Tensor mean");

			if (m_shape[nAxis] == 0)
				throw CommonLib::CExcBase("Tensor mean: nothing to reduce along axis {0}, size {1}", nAxis, ShapeText(m_shape));

			const TTensor<T> sums = Sum(nAxis);
			const TMeanValue count = TMeanValue(m_shape[nAxis]);

			TTensor<TMeanValue> result(sums.GetShape(), m_pAlloc);
			for (size_t i = 0; i < sums.Size(); ++i)
				result.Data()[i] = TMeanValue(sums.Data()[i]) / count;

			return result;
		}

		template<class T>
		TTensor<T> TTensor<T>::Max(size_t nAxis) const
		{
			CheckAxis(nAxis, "Tensor max");

			if (m_shape[nAxis] == 0)
				throw CommonLib::CExcBase("Tensor max: nothing to reduce along axis {0}, size {1}", nAxis, ShapeText(m_shape));

			const SAxis axis(m_shape, nAxis);

			TTensor<T> result(axis.reduced, m_pAlloc);
			for (size_t nOuter = 0; nOuter < axis.nOuter; ++nOuter)
			{
				TValue* pResult = result.m_pData + nOuter * axis.nInner;
				for (size_t nStep = 0; nStep < axis.nLength; ++nStep)
				{
					const TValue* pSource = m_pData + (nOuter * axis.nLength + nStep) * axis.nInner;
					for (size_t nInner = 0; nInner < axis.nInner; ++nInner)
					{
						if (nStep == 0 || pSource[nInner] > pResult[nInner])
							pResult[nInner] = pSource[nInner];
					}
				}
			}

			return result;
		}

		// ---- Reordering the axes ----

		template<class T>
		TTensor<T> TTensor<T>::Permute(const TTensor<T>& tensor, const TShape& axes)
		{
			const size_t nRank = tensor.Rank();

			std::vector<bool> used(nRank, false);
			bool bValid = axes.size() == nRank;
			for (size_t i = 0; i < axes.size() && bValid; ++i)
			{
				bValid = axes[i] < nRank && !used[axes[i]];
				if (bValid)
					used[axes[i]] = true;
			}

			if (!bValid)
			{
				throw CommonLib::CExcBase("Tensor permute: the axes ({0}) are not an order of the {1} axes of the tensor",
					Join(axes, ", "), nRank);
			}

			// How far apart, in the buffer of 'tensor', are two neighbors along each of its axes.
			std::vector<size_t> strides(nRank, 1);
			for (size_t i = nRank; i-- > 1;)
				strides[i - 1] = strides[i] * tensor.m_shape[i];

			TShape shape(nRank);
			std::vector<size_t> steps(nRank);	// the same for the axes of the result
			for (size_t i = 0; i < nRank; ++i)
			{
				shape[i] = tensor.m_shape[axes[i]];
				steps[i] = strides[axes[i]];
			}

			TTensor<T> result(shape, tensor.m_pAlloc);

			// Walk the result in its own order, like an odometer over its indexes, and
			// keep the matching position in 'tensor'.
			std::vector<size_t> indexes(nRank, 0);
			size_t nSource = 0;
			for (size_t n = 0; n < result.m_nSize; ++n)
			{
				result.m_pData[n] = tensor.m_pData[nSource];

				for (size_t i = nRank; i-- > 0;)
				{
					nSource += steps[i];
					if (++indexes[i] < shape[i])
						break;

					nSource -= steps[i] * shape[i];	// this axis wraps around, the next one moves
					indexes[i] = 0;
				}
			}

			return result;
		}

		// ---- Vectors and matrices ----

		template<class T>
		TTensor<T> TTensor<T>::FromVector(const TVector<T>& vec)
		{
			TTensor<T> result(TShape{vec.Size()}, vec.GetAlloc());
			if (!result.Empty())
				memcpy(result.m_pData, vec.Data(), result.m_nSize * sizeof(TValue));

			return result;
		}

		template<class T>
		TTensor<T> TTensor<T>::FromMatrix(const TMatrix<T>& matrix)
		{
			TTensor<T> result(TShape{matrix.Rows(), matrix.Cols()}, matrix.GetAlloc());
			if (!result.Empty())
				memcpy(result.m_pData, matrix.Data(), result.m_nSize * sizeof(TValue));

			return result;
		}

		template<class T>
		TVector<T> TTensor<T>::ToVector() const
		{
			if (Rank() != 1)
				throw CommonLib::CExcBase("Tensor to vector: the tensor has {0} axes, size {1}; reshape it to one axis first", Rank(), ShapeText(m_shape));

			TVector<T> result(m_nSize, m_pAlloc);
			if (!Empty())
				memcpy(result.Data(), m_pData, m_nSize * sizeof(TValue));

			return result;
		}

		template<class T>
		TMatrix<T> TTensor<T>::ToMatrix() const
		{
			if (Rank() != 2)
				throw CommonLib::CExcBase("Tensor to matrix: the tensor has {0} axes, size {1}; reshape it to two axes first", Rank(), ShapeText(m_shape));

			TMatrix<T> result(m_shape[0], m_shape[1], m_pAlloc);
			if (!Empty())
				memcpy(result.Data(), m_pData, m_nSize * sizeof(TValue));

			return result;
		}

		// ---- Helpers ----

		// Allocates the buffer of an empty tensor; the elements are not initialized.
		template<class T>
		void TTensor<T>::Create(const TShape& shape)
		{
			assert(m_pData == nullptr);

			const size_t nSize = ElementCount(shape, sizeof(TValue));

			// A tensor with an axis of size 0 has no buffer (IAlloc::Alloc rejects size 0).
			if (nSize != 0)
				m_pData = (TValue*)m_pAlloc->Alloc(nSize * sizeof(TValue));

			m_shape = shape;
			m_nSize = nSize;
		}

		template<class T>
		void TTensor<T>::CheckSameShape(const TTensor<T>& tensor, const char* pszName) const
		{
			if (m_shape != tensor.m_shape)
				throw CommonLib::CExcBase("{0}: sizes {1} and {2} do not match", std::string(pszName), ShapeText(m_shape), ShapeText(tensor.m_shape));
		}

		template<class T>
		void TTensor<T>::CheckAxis(size_t nAxis, const char* pszName) const
		{
			if (nAxis >= m_shape.size())
				throw CommonLib::CExcBase("{0}: there is no axis {1}, the tensor has {2} axes", std::string(pszName), nAxis, m_shape.size());
		}

		template<class T>
		size_t TTensor<T>::CheckedOffset(const TShape& indexes) const
		{
			bool bValid = indexes.size() == m_shape.size() && !indexes.empty();

			size_t nOffset = 0;
			for (size_t i = 0; i < indexes.size() && bValid; ++i)
			{
				bValid = indexes[i] < m_shape[i];
				if (bValid)
					nOffset = nOffset * m_shape[i] + indexes[i];
			}

			if (!bValid)
				throw CommonLib::CExcBase("Tensor: element ({0}) is out of range, size {1}", Join(indexes, ", "), ShapeText(m_shape));

			return nOffset;
		}

		// The element types the library supports.
		template class TTensor<int>;
		template class TTensor<float>;
		template class TTensor<double>;
	}
}
