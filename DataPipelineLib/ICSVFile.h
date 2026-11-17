#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "NumericsCoreLib/Matrix.h"
#include "NumericsCoreLib/Vector.h"
#include "CSVConverter.h"

namespace ML
{
	namespace data
	{
		struct SCSVOptions
		{
			// Field separator: ',' by default, ';' or '\t' for other dialects.
			char delimiter = ',';
			// True: the first line holds the column names.
			// False: the first line is data and the columns are named "0", "1", ...
			bool hasHeader = true;
		};

		// A CSV file loaded in memory: the column names and the cells as text.
		// CreateMatrix turns the columns the caller chooses into a matrix, CreateMatrixDrop
		// all the columns except the chosen ones, CreateVector one column into a vector.
		class ICSVFile
		{
		public:
			virtual ~ICSVFile() = default;

			// Column names, in file order.
			virtual const std::vector<std::string>& GetColumns() const = 0;
			// Number of data rows (the header line is not counted).
			virtual size_t GetRowCount() const = 0;
			// Index of the column in GetColumns(). Throws CommonLib::CExcBase when there
			// is no such column.
			virtual size_t GetColumnIndex(const std::string& column) const = 0;
			// The text of a cell. Throws CommonLib::CExcBase when it is out of range.
			virtual const std::string& GetCell(size_t nRow, size_t nColumn) const = 0;

			// Fills 'matrix' with GetRowCount() x columns.size() elements: matrix column j
			// holds the file column columns[j], so columns can be picked and reordered.
			// The element type is the type of the matrix; the buffer comes from the
			// allocator of the matrix. Each cell goes through 'converter'; the default
			// one reads numbers (see TCSVNumberConverter).
			// Throws CommonLib::CExcBase when a column does not exist or the converter
			// rejects a cell; 'matrix' is then left unchanged.
			template<class T>
			void CreateMatrix(const std::vector<std::string>& columns, numerics::TMatrix<T>& matrix,
				const ICSVConverter<T>& converter = TCSVNumberConverter<T>()) const
			{
				std::vector<size_t> indexes;
				indexes.reserve(columns.size());
				for (const std::string& column : columns)
					indexes.push_back(GetColumnIndex(column));

				FillMatrix(indexes, matrix, converter);
			}

			// The opposite choice: fills 'matrix' with all the columns of the file except
			// 'dropedColumns', in file order. Typical use: everything but the target and
			// the columns that are not features (an id, a name). Otherwise the same as
			// CreateMatrix; a dropped column that does not exist throws too.
			template<class T>
			void CreateMatrixDrop(const std::vector<std::string>& dropedColumns, numerics::TMatrix<T>& matrix,
				const ICSVConverter<T>& converter = TCSVNumberConverter<T>()) const
			{
				std::vector<bool> droped(GetColumns().size(), false);
				for (const std::string& column : dropedColumns)
					droped[GetColumnIndex(column)] = true;

				std::vector<size_t> indexes;
				for (size_t i = 0; i < droped.size(); ++i)
				{
					if (!droped[i])
						indexes.push_back(i);
				}

				FillMatrix(indexes, matrix, converter);
			}

			// Fills 'vec' with the GetRowCount() values of one column: the usual way to
			// load the target (y). The element type, the allocator, the converter and the
			// errors are the same as in CreateMatrix.
			template<class T>
			void CreateVector(const std::string& column, numerics::TVector<T>& vec,
				const ICSVConverter<T>& converter = TCSVNumberConverter<T>()) const
			{
				const size_t nColumn = GetColumnIndex(column);
				const size_t nRows = GetRowCount();

				numerics::TVector<T> result(nRows, vec.GetAlloc());
				for (size_t i = 0; i < nRows; ++i)
					result[i] = converter.Convert(GetCell(i, nColumn), column, i);

				vec.Swap(result);
			}

		private:
			// Fills 'matrix' with the file columns 'indexes', one matrix column each.
			template<class T>
			void FillMatrix(const std::vector<size_t>& indexes, numerics::TMatrix<T>& matrix, const ICSVConverter<T>& converter) const
			{
				const std::vector<std::string>& columns = GetColumns();
				const size_t nRows = GetRowCount();

				numerics::TMatrix<T> result(nRows, indexes.size(), matrix.GetAlloc());
				for (size_t i = 0; i < nRows; ++i)
				{
					for (size_t j = 0; j < indexes.size(); ++j)
						result(i, j) = converter.Convert(GetCell(i, indexes[j]), columns[indexes[j]], i);
				}

				matrix.Swap(result);
			}
		};

		typedef std::shared_ptr<ICSVFile> ICSVFilePtr;
	}
}
