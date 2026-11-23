#include "PolynomialFeatures.h"

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace data
	{
		using numerics::TMatrix;

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
						throw CommonLib::CExcBase("Polynomial features: more than {0} terms; lower the degree", c_nMaxTerms);

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
		TPolynomialFeatures<T>::TPolynomialFeatures(size_t nDegree) : m_nDegree(nDegree)
		{
			if (nDegree == 0)
				throw CommonLib::CExcBase("Polynomial features: the degree must be at least 1");
		}

		template<class T>
		void TPolynomialFeatures<T>::Fit(const TMatrix<T>& x)
		{
			m_bFitted = false;
			m_terms = Terms(x.Cols(), m_nDegree);
			m_nFeatures = x.Cols();
			m_bFitted = true;
		}

		template<class T>
		TMatrix<T> TPolynomialFeatures<T>::Transform(const TMatrix<T>& x) const
		{
			if (!m_bFitted)
				throw CommonLib::CExcBase("Polynomial features: the object is not fitted");

			if (x.Cols() != m_nFeatures)
				throw CommonLib::CExcBase("Polynomial features: the data has {0} columns, the object was fitted with {1}", x.Cols(), m_nFeatures);

			return Features(x, m_terms);
		}

		template<class T>
		TMatrix<T> TPolynomialFeatures<T>::FitTransform(const TMatrix<T>& x)
		{
			Fit(x);
			return Transform(x);
		}

		template<class T>
		std::vector<typename TPolynomialFeatures<T>::TTerm> TPolynomialFeatures<T>::Terms(size_t nFeatures, size_t nDegree)
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
		TMatrix<T> TPolynomialFeatures<T>::Features(const TMatrix<T>& x, const std::vector<TTerm>& terms)
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
		template class TPolynomialFeatures<float>;
		template class TPolynomialFeatures<double>;
	}
}
