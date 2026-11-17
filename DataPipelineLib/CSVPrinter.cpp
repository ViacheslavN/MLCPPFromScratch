#include "CSVPrinter.h"

#include "CommonLib/CommonLib.h"
#include "CommonLib/exception/exc_base.h"
#include "CommonLib/log/logger.h"

#include <cstdio>

namespace ML
{
	namespace data
	{
		namespace
		{
			const size_t c_nMaxCellLength = 30;

			// The number of characters of a UTF-8 text: the bytes that start a character.
			size_t Length(const std::string& text)
			{
				size_t nLength = 0;
				for (char ch : text)
				{
					if ((static_cast<unsigned char>(ch) & 0xC0) != 0x80)
						++nLength;
				}

				return nLength;
			}

			// A cell as it is printed: on one line and not longer than c_nMaxCellLength.
			std::string Printable(const std::string& text)
			{
				std::string result;
				size_t nLength = 0;

				for (size_t i = 0; i < text.size(); ++i)
				{
					const char ch = text[i];
					const bool bCharacterStart = (static_cast<unsigned char>(ch) & 0xC0) != 0x80;

					if (bCharacterStart && nLength == c_nMaxCellLength - 3 && Length(text) > c_nMaxCellLength)
					{
						result += "...";
						break;
					}

					if (bCharacterStart)
						++nLength;

					result += (ch == '\r' || ch == '\n' || ch == '\t') ? ' ' : ch;
				}

				return result;
			}

			// 'text' with spaces in front (bRight) or behind, up to nWidth characters.
			std::string Pad(const std::string& text, size_t nWidth, bool bRight)
			{
				const size_t nLength = Length(text);
				const std::string spaces(nWidth > nLength ? nWidth - nLength : 0, ' ');

				return bRight ? spaces + text : text + spaces;
			}

			const ICSVFile& Dereference(const ICSVFilePtr& pCsv)
			{
				if (pCsv.get() == nullptr)
					throw CommonLib::CExcBase("CSV print: the file is null");

				return *pCsv;
			}
		}

		void CCSVPrinter::Head(const ICSVFile& csv, size_t nRows)
		{
			LogLines(HeadLines(csv, nRows));
		}

		void CCSVPrinter::Head(const ICSVFilePtr& pCsv, size_t nRows)
		{
			Head(Dereference(pCsv), nRows);
		}

		void CCSVPrinter::Row(const ICSVFile& csv, size_t nRow)
		{
			LogLines(RowLines(csv, nRow));
		}

		void CCSVPrinter::Row(const ICSVFilePtr& pCsv, size_t nRow)
		{
			Row(Dereference(pCsv), nRow);
		}

		std::vector<std::string> CCSVPrinter::HeadLines(const ICSVFile& csv, size_t nRows)
		{
			const std::vector<std::string>& columns = csv.GetColumns();
			const size_t nShown = (std::min)(nRows, csv.GetRowCount());

			std::vector<std::string> cells;
			cells.reserve(nShown * columns.size());
			for (size_t i = 0; i < nShown; ++i)
			{
				for (size_t j = 0; j < columns.size(); ++j)
					cells.push_back(csv.GetCell(i, j));
			}

			return TableLines(csv.GetRowCount(), columns, cells);
		}

		std::vector<std::string> CCSVPrinter::RowLines(const ICSVFile& csv, size_t nRow)
		{
			const std::vector<std::string>& columns = csv.GetColumns();
			CheckRow(nRow, csv.GetRowCount());

			std::vector<std::string> cells;
			cells.reserve(columns.size());
			for (size_t j = 0; j < columns.size(); ++j)
				cells.push_back(csv.GetCell(nRow, j));

			return RecordLines(nRow, csv.GetRowCount(), columns, cells);
		}

		std::vector<std::string> CCSVPrinter::TableLines(size_t nTotalRows, const std::vector<std::string>& columns,
			const std::vector<std::string>& cells)
		{
			const size_t nCols = columns.size();
			const size_t nShown = nCols == 0 ? 0 : cells.size() / nCols;

			std::vector<std::string> lines;
			lines.push_back(std::to_string(nTotalRows) + " rows x " + std::to_string(nCols) + " columns"
				+ (nShown < nTotalRows && nCols != 0 ? ", the first " + std::to_string(nShown) + ":" : ":"));

			if (nCols == 0)
				return lines;

			// The table as text: the names, then the shown rows.
			std::vector<std::vector<std::string>> table(nShown + 1, std::vector<std::string>(nCols));
			std::vector<size_t> widths(nCols, 0);

			for (size_t j = 0; j < nCols; ++j)
			{
				table[0][j] = Printable(columns[j]);
				for (size_t i = 0; i < nShown; ++i)
					table[i + 1][j] = Printable(cells[i * nCols + j]);

				for (size_t i = 0; i <= nShown; ++i)
					widths[j] = (std::max)(widths[j], Length(table[i][j]));
			}

			// The row numbers take the width of the largest one shown.
			const size_t nNumberWidth = std::to_string(nShown == 0 ? 0 : nShown - 1).size();

			for (size_t i = 0; i <= nShown; ++i)
			{
				std::string line = Pad(i == 0 ? std::string() : std::to_string(i - 1), nNumberWidth, true);
				for (size_t j = 0; j < nCols; ++j)
					line += "  " + Pad(table[i][j], widths[j], true);

				lines.push_back(line);
			}

			return lines;
		}

		std::vector<std::string> CCSVPrinter::RecordLines(size_t nRow, size_t nTotalRows, const std::vector<std::string>& columns,
			const std::vector<std::string>& cells)
		{
			std::vector<std::string> names(columns.size());
			size_t nWidth = 0;
			for (size_t j = 0; j < columns.size(); ++j)
			{
				names[j] = Printable(columns[j]);
				nWidth = (std::max)(nWidth, Length(names[j]));
			}

			std::vector<std::string> lines;
			lines.push_back("row " + std::to_string(nRow) + " of " + std::to_string(nTotalRows) + ":");
			for (size_t j = 0; j < columns.size(); ++j)
				lines.push_back("  " + Pad(names[j], nWidth, false) + " : " + Printable(cells[j]));

			return lines;
		}

		std::vector<std::string> CCSVPrinter::ColumnNames(size_t nCols, const std::vector<std::string>& columns)
		{
			if (!columns.empty())
			{
				if (columns.size() != nCols)
					throw CommonLib::CExcBase("Print: {0} column names for a matrix of {1} columns", columns.size(), nCols);

				return columns;
			}

			std::vector<std::string> names(nCols);
			for (size_t j = 0; j < nCols; ++j)
				names[j] = std::to_string(j);

			return names;
		}

		void CCSVPrinter::CheckRow(size_t nRow, size_t nTotalRows)
		{
			if (nRow >= nTotalRows)
				throw CommonLib::CExcBase("Print: there is no row {0}, there are {1} rows", nRow, nTotalRows);
		}

		void CCSVPrinter::LogLines(const std::vector<std::string>& lines)
		{
			// Info(msg) with one argument does not treat the text as a format, so
			// braces in the data are printed as they are.
			for (const std::string& line : lines)
				Log.Info(line);
		}

		std::string CCSVPrinter::ToText(double value)
		{
			// %g: up to 6 significant digits, no trailing zeros, an exponent when needed.
			char buffer[32];
			snprintf(buffer, sizeof(buffer), "%g", value);
			return buffer;
		}

		std::string CCSVPrinter::ToText(float value)
		{
			return ToText(double(value));
		}

		std::string CCSVPrinter::ToText(int value)
		{
			return std::to_string(value);
		}
	}
}
