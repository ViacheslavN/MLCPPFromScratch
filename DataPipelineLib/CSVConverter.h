#pragma once

#include <cstddef>
#include <string>
#include <type_traits>

namespace ML
{
	namespace data
	{
		// Turns the text of a CSV cell into a matrix element of type T. Passed to
		// ICSVFile::CreateMatrix; implement it to load data that is not plain numbers
		// (categories, dates, yes/no, numbers with a decimal comma).
		template<class T>
		class ICSVConverter
		{
		public:
			virtual ~ICSVConverter() = default;

			// 'text': the cell as it is in the file (quotes already removed).
			// 'column': the name of its column, 'nRow': its data row, from 0. A converter
			// can use them to treat columns differently and for its error messages.
			virtual T Convert(const std::string& text, const std::string& column, size_t nRow) const = 0;
		};

		// The default converter: the cell is a number. T is int, float or double (the
		// member functions are compiled for these three in CSVConverter.cpp).
		//
		// Spaces and tabs around the number are ignored; the decimal separator is always
		// '.', whatever the system locale. Text that is not a number of type T (for int
		// this includes "1.5" and values out of range) throws CommonLib::CExcBase.
		template<class T>
		class TCSVNumberConverter : public ICSVConverter<T>
		{
		public:
			// An empty cell (a missing value) becomes NaN for float and double, and
			// throws CommonLib::CExcBase for int, which has no NaN.
			TCSVNumberConverter();
			// An empty cell becomes 'missingValue'.
			explicit TCSVNumberConverter(T missingValue);

			T Convert(const std::string& text, const std::string& column, size_t nRow) const override;

		private:
			bool m_bHasMissingValue;
			T m_missingValue;
		};
	}
}
