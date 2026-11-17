#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ICSVFile.h"
#include "NumericsCoreLib/Matrix.h"

namespace ML
{
	namespace data
	{
		// Prints the content of a CSV file, or of a matrix made from it, to the CommonLib
		// log, to look at the data before working with it. Static methods only.
		//
		// Every line goes to the global CommonLib logger as an info message
		// (Log.Info). The logger writes nothing until it is set up with Log.InnitLog,
		// for example with a CConsoleLogWriter to print to the console.
		//
		// Long cells are cut to 30 characters and line breaks inside a cell are shown
		// as spaces, so that every row stays on one line. Numbers of a matrix are
		// printed with up to 6 significant digits.
		class CCSVPrinter
		{
		public:
			// ---- CSV file ----

			// The first 'nRows' rows as a table: the column names, then one line per
			// row, starting with the row number (from 0). The columns are aligned.
			// Fewer rows are printed when the file has fewer.
			static void Head(const ICSVFile& csv, size_t nRows = 5);
			static void Head(const ICSVFilePtr& pCsv, size_t nRows = 5);

			// One row, a line per column: "name : value". For files with too many
			// columns to read as a table. 'nRow' is the row number, from 0; throws
			// CommonLib::CExcBase when the file has no such row.
			static void Row(const ICSVFile& csv, size_t nRow);
			static void Row(const ICSVFilePtr& pCsv, size_t nRow);

			// The same text as lines, without logging it.
			static std::vector<std::string> HeadLines(const ICSVFile& csv, size_t nRows = 5);
			static std::vector<std::string> RowLines(const ICSVFile& csv, size_t nRow);

			// ---- Matrix ----

			// The same two views for a matrix of any element type (int, float, double).
			// 'columns' are the names to print above the columns, one per column of the
			// matrix: usually the names the matrix was created with. Without them the
			// columns are numbered 0, 1, ... Throws CommonLib::CExcBase when the number
			// of names is not the number of columns.

			// The first 'nRows' rows as a table.
			template<class T>
			static void Head(const numerics::TMatrix<T>& matrix, size_t nRows = 5,
				const std::vector<std::string>& columns = std::vector<std::string>())
			{
				LogLines(HeadLines(matrix, nRows, columns));
			}

			// The row 'nRow' (from 0), a line per column.
			template<class T>
			static void Row(const numerics::TMatrix<T>& matrix, size_t nRow,
				const std::vector<std::string>& columns = std::vector<std::string>())
			{
				LogLines(RowLines(matrix, nRow, columns));
			}

			template<class T>
			static std::vector<std::string> HeadLines(const numerics::TMatrix<T>& matrix, size_t nRows = 5,
				const std::vector<std::string>& columns = std::vector<std::string>())
			{
				const size_t nShown = nRows < matrix.Rows() ? nRows : matrix.Rows();

				std::vector<std::string> cells;
				cells.reserve(nShown * matrix.Cols());
				for (size_t i = 0; i < nShown; ++i)
				{
					for (size_t j = 0; j < matrix.Cols(); ++j)
						cells.push_back(ToText(matrix(i, j)));
				}

				return TableLines(matrix.Rows(), ColumnNames(matrix.Cols(), columns), cells);
			}

			template<class T>
			static std::vector<std::string> RowLines(const numerics::TMatrix<T>& matrix, size_t nRow,
				const std::vector<std::string>& columns = std::vector<std::string>())
			{
				const std::vector<std::string> names = ColumnNames(matrix.Cols(), columns);
				CheckRow(nRow, matrix.Rows());

				std::vector<std::string> cells;
				cells.reserve(matrix.Cols());
				for (size_t j = 0; j < matrix.Cols(); ++j)
					cells.push_back(ToText(matrix(nRow, j)));

				return RecordLines(nRow, matrix.Rows(), names, cells);
			}

		private:
			// The table of the first rows: 'cells' holds the shown rows, row by row.
			static std::vector<std::string> TableLines(size_t nTotalRows, const std::vector<std::string>& columns,
				const std::vector<std::string>& cells);
			// One row, a line per column: 'cells' holds its values.
			static std::vector<std::string> RecordLines(size_t nRow, size_t nTotalRows, const std::vector<std::string>& columns,
				const std::vector<std::string>& cells);
			// The names for nCols columns: 'columns', or "0", "1", ... when it is empty.
			static std::vector<std::string> ColumnNames(size_t nCols, const std::vector<std::string>& columns);
			static void CheckRow(size_t nRow, size_t nTotalRows);
			static void LogLines(const std::vector<std::string>& lines);

			static std::string ToText(double value);
			static std::string ToText(float value);
			static std::string ToText(int value);
		};
	}
}
