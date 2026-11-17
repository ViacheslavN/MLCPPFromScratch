#include "Model.h"
#include "ModelUtils.h"

#include <string>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace supervised
	{
		namespace
		{
			// A vector of feature values as a matrix with one column.
			template<class T>
			TMatrix<T> AsColumn(const TVector<T>& x)
			{
				TMatrix<T> column(x.Size(), 1, x.GetAlloc());
				for (size_t i = 0; i < x.Size(); ++i)
					column(i, 0) = x[i];

				return column;
			}
		}

		// ---- IRegressorT ----

		template<class T>
		void IRegressorT<T>::Fit(const TVector<T>& x, const TVector<T>& y)
		{
			Fit(AsColumn(x), y);
		}

		template<class T>
		TVector<T> IRegressorT<T>::Predict(const TVector<T>& x) const
		{
			return Predict(AsColumn(x));
		}

		template<class T>
		double IRegressorT<T>::Score(const TVector<T>& x, const TVector<T>& y) const
		{
			return Score(AsColumn(x), y);
		}

		template<class T>
		double IRegressorT<T>::Score(const TMatrix<T>& x, const TVector<T>& y) const
		{
			const TVector<T> predicted = Predict(x);
			if (predicted.Size() != y.Size())
				throw CommonLib::CExcBase("Score: {0} rows and {1} target values", predicted.Size(), y.Size());

			if (y.Empty())
				throw CommonLib::CExcBase("Score: no samples");

			// The score is computed in double whatever T is.
			double mean = 0;
			for (size_t i = 0; i < y.Size(); ++i)
				mean += double(y[i]);

			mean /= double(y.Size());

			double residual = 0;	// sum of squared errors of the model
			double total = 0;		// the same for the constant model "always the mean"
			for (size_t i = 0; i < y.Size(); ++i)
			{
				const double error = double(y[i]) - double(predicted[i]);
				residual += error * error;
				total += (double(y[i]) - mean) * (double(y[i]) - mean);
			}

			// All targets are equal: R^2 is not defined; 1 for exact predictions, else 0.
			if (total == 0)
				return residual == 0 ? 1.0 : 0.0;

			return 1 - residual / total;
		}

		template class IRegressorT<float>;
		template class IRegressorT<double>;

		// ---- IClassifier ----

		void IClassifier::Fit(const CVector& x, const CVectorI& y)
		{
			Fit(AsColumn(x), y);
		}

		CVectorI IClassifier::Predict(const CVector& x) const
		{
			return Predict(AsColumn(x));
		}

		double IClassifier::Score(const CVector& x, const CVectorI& y) const
		{
			return Score(AsColumn(x), y);
		}

		double IClassifier::Score(const CMatrix& x, const CVectorI& y) const
		{
			const CVectorI predicted = Predict(x);
			if (predicted.Size() != y.Size())
				throw CommonLib::CExcBase("Score: {0} rows and {1} labels", predicted.Size(), y.Size());

			if (y.Empty())
				throw CommonLib::CExcBase("Score: no samples");

			size_t nCorrect = 0;
			for (size_t i = 0; i < y.Size(); ++i)
			{
				if (predicted[i] == y[i])
					++nCorrect;
			}

			return double(nCorrect) / double(y.Size());
		}

		namespace detail
		{
			void CheckFit(size_t nRows, size_t nCols, size_t nTargets, const char* pszModel)
			{
				if (nRows == 0 || nCols == 0)
					throw CommonLib::CExcBase("{0}: no data to fit, size {1} x {2}", std::string(pszModel), nRows, nCols);

				if (nRows != nTargets)
					throw CommonLib::CExcBase("{0}: {1} samples and {2} targets", std::string(pszModel), nRows, nTargets);
			}

			void CheckPredict(size_t nCols, size_t nFeatures, bool bFitted, const char* pszModel)
			{
				if (!bFitted)
					throw CommonLib::CExcBase("{0}: the model is not fitted", std::string(pszModel));

				if (nCols != nFeatures)
					throw CommonLib::CExcBase("{0}: the data has {1} features, the model was fitted with {2}", std::string(pszModel), nCols, nFeatures);
			}

			size_t CountClasses(const CVectorI& y, const char* pszModel)
			{
				int nMax = 0;
				for (size_t i = 0; i < y.Size(); ++i)
				{
					if (y[i] < 0)
						throw CommonLib::CExcBase("{0}: label {1} of sample {2} is negative, the classes are numbered from 0", std::string(pszModel), y[i], i);

					if (y[i] > nMax)
						nMax = y[i];
				}

				if (nMax == 0)
					throw CommonLib::CExcBase("{0}: all samples have class 0, at least two classes are needed", std::string(pszModel));

				return size_t(nMax) + 1;
			}

			size_t ArgMax(const double* pValues, size_t nCount)
			{
				size_t nMax = 0;
				for (size_t i = 1; i < nCount; ++i)
				{
					if (pValues[i] > pValues[nMax])
						nMax = i;
				}

				return nMax;
			}
		}
	}
}
