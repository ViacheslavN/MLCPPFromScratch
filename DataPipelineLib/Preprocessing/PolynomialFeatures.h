#pragma once

#include <cstddef>
#include <vector>

#include "NumericsCoreLib/Matrix.h"

namespace ML
{
	namespace data
	{
		// Polynomial features: new columns made of powers and products of the features.
		// For two features x0, x1 and degree 2 the result has the columns
		//   x0, x1, x0^2, x0 * x1, x1^2.
		// A linear model fitted on them is a polynomial of the original features, so it
		// can follow curves and interactions between features that a straight line
		// cannot. There is no column of ones: the models fit the intercept themselves.
		//
		// The number of columns grows fast: 5 for two features of degree 2, 55 for ten
		// features of degree 2, 285 for ten of degree 3. High powers also make the
		// columns nearly dependent: keep the features in a small range (about -1..1,
		// see TStandardScaler) for degrees above 5 or so, more so with float.
		//
		// Fit learns the number of features, Transform builds the columns. Nothing is
		// learned from the values, so the same object serves training and test data.
		//
		// T is float or double (the member functions are compiled for these two in
		// PolynomialFeatures.cpp). CPolynomialFeatures is the double version,
		// CPolynomialFeaturesF the float one. Wrong sizes and a Transform before a Fit
		// throw CommonLib::CExcBase.
		template<class T>
		class TPolynomialFeatures
		{
		public:
			// One term of the polynomial: the power of every feature in it.
			// {2, 0} is x0^2, {1, 1} is x0 * x1.
			typedef std::vector<size_t> TTerm;

			// 'nDegree' >= 1: the largest total power of a term. 1 leaves the features
			// as they are.
			explicit TPolynomialFeatures(size_t nDegree = 2);

			// Takes the number of features from the columns of 'x' and prepares the
			// terms. Throws when there would be more than 100000 of them.
			void Fit(const numerics::TMatrix<T>& x);
			// Rows() x GetTerms().size(): the value of every term for every row of 'x',
			// which must have the columns of the Fit. The result uses the allocator of 'x'.
			numerics::TMatrix<T> Transform(const numerics::TMatrix<T>& x) const;
			// Fit(x), then Transform(x).
			numerics::TMatrix<T> FitTransform(const numerics::TMatrix<T>& x);

			size_t GetDegree() const { return m_nDegree; }
			// The terms, one per column of the result, in its order: by total power,
			// and inside one power from the terms with the most of x0 down.
			const std::vector<TTerm>& GetTerms() const { return m_terms; }
			bool IsFitted() const { return m_bFitted; }

			// The same without an object.
			// The terms of a polynomial of 'nFeatures' features up to 'nDegree'.
			static std::vector<TTerm> Terms(size_t nFeatures, size_t nDegree);
			// The value of every term of 'terms' for every row of 'x'.
			static numerics::TMatrix<T> Features(const numerics::TMatrix<T>& x, const std::vector<TTerm>& terms);

		private:
			size_t m_nDegree;
			std::vector<TTerm> m_terms;
			size_t m_nFeatures = 0;
			bool m_bFitted = false;
		};

		typedef TPolynomialFeatures<double> CPolynomialFeatures;
		typedef TPolynomialFeatures<float> CPolynomialFeaturesF;
	}
}
