#include "PolynomialRegression.h"
#include "LinearRegression.h"
#include "ModelUtils.h"

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace supervised
	{
		namespace
		{
			// More terms than this is a mistake in the degree or in the number of features.
			const size_t c_nMaxTerms = 100000;

			// Adds all the terms where the features from nFeature on share 'nPower'.
			void AddTerms(std::vector<std::vector<size_t>>& terms, std::vector<size_t>& term, size_t nFeature, size_t nPower)
			{
				if (nFeature + 1 == term.size())
				{
					term[nFeature] = nPower;	// the last feature takes what is left
					terms.push_back(term);

					if (terms.size() > c_nMaxTerms)
						throw CommonLib::CExcBase("Polynomial regression: more than {0} terms; lower the degree", c_nMaxTerms);

					return;
				}

				for (size_t n = nPower + 1; n-- > 0;)
				{
					term[nFeature] = n;
					AddTerms(terms, term, nFeature + 1, nPower - n);
				}
			}
		}

		template<class T>
		TPolynomialRegression<T>::TPolynomialRegression(size_t nDegree, T alpha) : m_nDegree(nDegree), m_alpha(alpha)
		{
			if (nDegree == 0)
				throw CommonLib::CExcBase("Polynomial regression: the degree must be at least 1");

			if (!(alpha >= 0))
				throw CommonLib::CExcBase("Polynomial regression: alpha {0} must not be negative", alpha);
		}

		template<class T>
		void TPolynomialRegression<T>::Fit(const TMatrix<T>& x, const TVector<T>& y)
		{
			detail::CheckFit(x.Rows(), x.Cols(), y.Size(), "Polynomial regression");

			m_bFitted = false;
			m_terms = Terms(x.Cols(), m_nDegree);

			const TMatrix<T> features = Features(x, m_terms);

			try
			{
				if (m_alpha > 0)
				{
					TRidgeRegression<T> model(m_alpha);
					model.Fit(features, y);
					m_weights = model.GetWeights();
					m_intercept = model.GetIntercept();
				}
				else
				{
					TLinearRegression<T> model;
					model.Fit(features, y);
					m_weights = model.GetWeights();
					m_intercept = model.GetIntercept();
				}
			}
			catch (CommonLib::CExcBase& exc)
			{
				exc.AddMsgT("Polynomial regression: failed to fit {0} terms of degree {1} on {2} samples", m_terms.size(), m_nDegree, x.Rows());
				throw;
			}

			m_nFeatures = x.Cols();
			m_bFitted = true;
		}

		template<class T>
		TVector<T> TPolynomialRegression<T>::Predict(const TMatrix<T>& x) const
		{
			detail::CheckPredict(x.Cols(), m_nFeatures, m_bFitted, "Polynomial regression");

			TVector<T> result = TMatrix<T>::Multiply(Features(x, m_terms), m_weights);
			for (size_t i = 0; i < result.Size(); ++i)
				result[i] += m_intercept;

			return result;
		}

		template<class T>
		std::vector<typename TPolynomialRegression<T>::TTerm> TPolynomialRegression<T>::Terms(size_t nFeatures, size_t nDegree)
		{
			std::vector<TTerm> terms;
			if (nFeatures == 0)
				return terms;

			TTerm term(nFeatures, 0);
			for (size_t nPower = 1; nPower <= nDegree; ++nPower)
				AddTerms(terms, term, 0, nPower);

			return terms;
		}

		template<class T>
		TMatrix<T> TPolynomialRegression<T>::Features(const TMatrix<T>& x, const std::vector<TTerm>& terms)
		{
			const size_t nCols = x.Cols();

			size_t nMaxPower = 0;
			for (const TTerm& term : terms)
			{
				if (term.size() != nCols)
					throw CommonLib::CExcBase("Polynomial features: a term has {0} powers, the data has {1} features", term.size(), nCols);

				for (size_t nPower : term)
					nMaxPower = (std::max)(nMaxPower, nPower);
			}

			TMatrix<T> result(x.Rows(), terms.size(), x.GetAlloc());

			// The powers 0..nMaxPower of every feature of the current row.
			std::vector<T> powers(nCols * (nMaxPower + 1));

			for (size_t i = 0; i < x.Rows(); ++i)
			{
				for (size_t j = 0; j < nCols; ++j)
				{
					T* pPowers = powers.data() + j * (nMaxPower + 1);
					pPowers[0] = 1;
					for (size_t k = 1; k <= nMaxPower; ++k)
						pPowers[k] = pPowers[k - 1] * x(i, j);
				}

				T* pResult = result.Data() + i * terms.size();
				for (size_t t = 0; t < terms.size(); ++t)
				{
					T value = 1;
					for (size_t j = 0; j < nCols; ++j)
						value *= powers[j * (nMaxPower + 1) + terms[t][j]];

					pResult[t] = value;
				}
			}

			return result;
		}

		// The element types the library supports.
		template class TPolynomialRegression<float>;
		template class TPolynomialRegression<double>;
	}
}
