#pragma once

// Shared includes and helpers for the NumericsCore Catch2 tests.

#ifdef _WIN32
#define NOMINMAX
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"
#include "NumericsCore/Matrix.h"
#include "NumericsCore/Vector.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace test_utils
{
	using CommonLib::CExcBase;
	using CommonLib::IAllocPtr;
	using ML::numerics::TMatrix;
	using ML::numerics::TVector;

	// An allocator that counts: GetCurrentMemoryBalance() is 0 when everything is freed.
	inline IAllocPtr MakeAlloc()
	{
		return CommonLib::IAlloc::CreateSimpleAlloc(true);
	}

	// Call when every matrix and vector of 'pAlloc' is destroyed.
	inline void RequireNoLeaks(const IAllocPtr& pAlloc)
	{
		REQUIRE(pAlloc->GetCurrentMemoryBalance() == 0);
		REQUIRE(pAlloc->GetAllocCount() == pAlloc->GetFreeCount());
	}

	// rows x cols matrix from its values, row by row.
	template<class T>
	TMatrix<T> MakeMatrix(size_t nRows, size_t nCols, std::initializer_list<double> values, IAllocPtr pAlloc = IAllocPtr())
	{
		REQUIRE(values.size() == nRows * nCols);

		TMatrix<T> matrix(nRows, nCols, pAlloc);
		size_t i = 0;
		for (double value : values)
			matrix.Data()[i++] = T(value);

		return matrix;
	}

	template<class T>
	TVector<T> MakeVector(std::initializer_list<double> values, IAllocPtr pAlloc = IAllocPtr())
	{
		TVector<T> vec(values.size(), pAlloc);
		size_t i = 0;
		for (double value : values)
			vec[i++] = T(value);

		return vec;
	}

	// Exact comparison with the expected values (row by row for a matrix).
	template<class T>
	bool Equals(const TMatrix<T>& matrix, std::initializer_list<double> values)
	{
		if (matrix.Size() != values.size())
			return false;

		size_t i = 0;
		for (double value : values)
		{
			if (matrix.Data()[i++] != T(value))
				return false;
		}

		return true;
	}

	template<class T>
	bool Equals(const TVector<T>& vec, std::initializer_list<double> values)
	{
		if (vec.Size() != values.size())
			return false;

		size_t i = 0;
		for (double value : values)
		{
			if (vec[i++] != T(value))
				return false;
		}

		return true;
	}

	// The largest difference between two matrices or vectors; huge when the sizes differ.
	template<class T>
	double MaxDiff(const TMatrix<T>& a, const TMatrix<T>& b)
	{
		if (a.Rows() != b.Rows() || a.Cols() != b.Cols())
			return std::numeric_limits<double>::max();

		double diff = 0;
		for (size_t i = 0; i < a.Size(); ++i)
			diff = std::max(diff, std::fabs(double(a.Data()[i]) - double(b.Data()[i])));

		return diff;
	}

	template<class T>
	double MaxDiff(const TVector<T>& a, const TVector<T>& b)
	{
		if (a.Size() != b.Size())
			return std::numeric_limits<double>::max();

		double diff = 0;
		for (size_t i = 0; i < a.Size(); ++i)
			diff = std::max(diff, std::fabs(double(a[i]) - double(b[i])));

		return diff;
	}

	template<class T>
	double MaxDiff(const TMatrix<T>& a, size_t nRows, size_t nCols, std::initializer_list<double> values)
	{
		return MaxDiff(a, MakeMatrix<T>(nRows, nCols, values));
	}

	template<class T>
	double MaxDiff(const TVector<T>& a, std::initializer_list<double> values)
	{
		return MaxDiff(a, MakeVector<T>(values));
	}

	// What counts as "equal up to rounding" for results of size about 1.
	template<class T>
	double Tolerance()
	{
		return std::is_same<T, float>::value ? 3e-4 : 1e-11;
	}

	// The same numbers on every platform (the std distributions differ between libraries).
	class CRandom
	{
	public:
		explicit CRandom(uint64_t seed = 12345) : m_seed(seed) {}

		// Uniform in [-1, 1).
		double Next()
		{
			m_seed = m_seed * 6364136223846793005ULL + 1442695040888963407ULL;
			return double(m_seed >> 11) / double(1ULL << 53) * 2 - 1;
		}

		template<class T>
		TMatrix<T> Matrix(size_t nRows, size_t nCols)
		{
			TMatrix<T> matrix(nRows, nCols);
			for (size_t i = 0; i < matrix.Size(); ++i)
				matrix.Data()[i] = T(Next());

			return matrix;
		}

		template<class T>
		TVector<T> Vector(size_t nSize)
		{
			TVector<T> vec(nSize);
			for (size_t i = 0; i < nSize; ++i)
				vec[i] = T(Next());

			return vec;
		}

	private:
		uint64_t m_seed;
	};

	template<class T>
	TMatrix<T> Identity(size_t nSize)
	{
		TMatrix<T> matrix(nSize, nSize);
		for (size_t i = 0; i < nSize; ++i)
			matrix(i, i) = 1;

		return matrix;
	}

	template<class T>
	TMatrix<T> Diagonal(const TVector<T>& values)
	{
		TMatrix<T> matrix(values.Size(), values.Size());
		for (size_t i = 0; i < values.Size(); ++i)
			matrix(i, i) = values[i];

		return matrix;
	}

	template<class T>
	TMatrix<T> Mul(const TMatrix<T>& a, const TMatrix<T>& b)
	{
		return TMatrix<T>::Multiply(a, b);
	}

	template<class T>
	TMatrix<T> Transposed(const TMatrix<T>& a)
	{
		return TMatrix<T>::Transpose(a);
	}
}
