#include "CSVConverter.h"

#include <charconv>
#include <limits>

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace data
	{
		template<class T>
		TCSVNumberConverter<T>::TCSVNumberConverter() : m_bHasMissingValue(std::is_floating_point<T>::value),
			m_missingValue(std::is_floating_point<T>::value ? std::numeric_limits<T>::quiet_NaN() : T(0))
		{}

		template<class T>
		TCSVNumberConverter<T>::TCSVNumberConverter(T missingValue) : m_bHasMissingValue(true), m_missingValue(missingValue)
		{}

		template<class T>
		T TCSVNumberConverter<T>::Convert(const std::string& text, const std::string& column, size_t nRow) const
		{
			// The cell without spaces and tabs at both ends.
			const char* pBegin = text.data();
			const char* pEnd = pBegin + text.size();
			while (pBegin < pEnd && (*pBegin == ' ' || *pBegin == '\t'))
				++pBegin;
			while (pEnd > pBegin && (pEnd[-1] == ' ' || pEnd[-1] == '\t'))
				--pEnd;

			if (pBegin == pEnd)
			{
				if (!m_bHasMissingValue)
					throw CommonLib::CExcBase("CSV: the value is empty (column '{0}', data row {1})", column, nRow + 1);

				return m_missingValue;
			}

			// std::from_chars does not depend on the locale and does not accept a leading '+'.
			const char* pNumber = *pBegin == '+' ? pBegin + 1 : pBegin;

			T value = 0;
			const std::from_chars_result result = std::from_chars(pNumber, pEnd, value);
			if (result.ec != std::errc() || result.ptr != pEnd)
			{
				throw CommonLib::CExcBase("CSV: '{0}' is not {1} (column '{2}', data row {3})", std::string(pBegin, pEnd),
					std::string(std::is_floating_point<T>::value ? "a number" : "an integer"), column, nRow + 1);
			}

			return value;
		}

		// The element types the library supports.
		template class TCSVNumberConverter<int>;
		template class TCSVNumberConverter<float>;
		template class TCSVNumberConverter<double>;
	}
}
