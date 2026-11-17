#pragma once

#include <vector>

#include "Model.h"

namespace ML
{
	namespace supervised
	{
		// Polynomial regression: a linear regression on powers and products of the
		// features. For one feature x and degree 3:
		//   prediction = intercept + w0 * x + w1 * x^2 + w2 * x^3.
		// With several features the products are included too; for x0, x1 and degree 2
		// the terms are x0, x1, x0^2, x0 * x1, x1^2. The model is still linear in its
		// weights, so it is fitted exactly, like a linear regression.
		//
		// A high degree follows the training data closely and behaves wildly between
		// and beyond the samples (overfitting); 'alpha' > 0 adds the ridge penalty that
		// keeps the weights small. High powers also make the columns nearly dependent:
		// keep the features in a small range (about -1..1) for degrees above 5 or so,
		// more so with float.
		//
		// T is float or double (the member functions are compiled for these two in
		// PolynomialRegression.cpp). CPolynomialRegression is the double version,
		// CPolynomialRegressionF the float one.
		template<class T>
		class TPolynomialRegression : public IRegressorT<T>
		{
		public:
			// One term of the polynomial: the power of every feature in it.
			// {2, 0} is x0^2, {1, 1} is x0 * x1.
			typedef std::vector<size_t> TTerm;

			// 'nDegree' >= 1: the largest total power of a term; 1 is a linear regression.
			// 'alpha' >= 0: the ridge penalty, alpha * sum of weights^2; 0 is none.
			explicit TPolynomialRegression(size_t nDegree = 2, T alpha = T(0));

			// Throws when the terms are linearly dependent and alpha is 0: too few
			// samples for the number of terms, or features that depend on each other.
			void Fit(const TMatrix<T>& x, const TVector<T>& y) override;
			using IRegressorT<T>::Fit;		// the overload for a vector of one feature
			TVector<T> Predict(const TMatrix<T>& x) const override;
			using IRegressorT<T>::Predict;

			size_t GetDegree() const { return m_nDegree; }
			// The terms, in the order of GetWeights(): by total power, and inside one
			// power from the terms with the most of x0 down.
			const std::vector<TTerm>& GetTerms() const { return m_terms; }
			// One weight per term.
			const TVector<T>& GetWeights() const { return m_weights; }
			T GetIntercept() const { return m_intercept; }

			// The terms of a polynomial of 'nFeatures' features up to 'nDegree'.
			static std::vector<TTerm> Terms(size_t nFeatures, size_t nDegree);
			// Rows() x terms.size(): the value of every term for every row of 'x'. This
			// is the matrix the linear regression is fitted on.
			static TMatrix<T> Features(const TMatrix<T>& x, const std::vector<TTerm>& terms);

		private:
			size_t m_nDegree;
			T m_alpha;
			std::vector<TTerm> m_terms;
			TVector<T> m_weights;
			T m_intercept = 0;
			size_t m_nFeatures = 0;
			bool m_bFitted = false;
		};

		typedef TPolynomialRegression<double> CPolynomialRegression;
		typedef TPolynomialRegression<float> CPolynomialRegressionF;
	}
}
