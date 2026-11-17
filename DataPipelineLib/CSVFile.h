#pragma once

#include "CommonLib/CommonLib.h"
#include "CommonLib/filesystem/File.h"

#include "ICSVFile.h"

namespace ML
{
	namespace data
	{
		// ICSVFile that reads the whole file through CommonLib::file::IFile.
		//
		// Format: fields separated by the delimiter, lines ended by LF, CRLF or CR.
		// A field in double quotes can contain the delimiter, line breaks and quotes
		// (written as ""). A UTF-8 byte order mark and blank lines are skipped. Every
		// line must have the same number of fields as the first one.
		class CCSVFile : public ICSVFile
		{
		public:
			// Reads 'pFile' from its beginning to its end; the file is not used after
			// the constructor. Throws CommonLib::CExcBase on a read or format error.
			CCSVFile(CommonLib::file::TFilePtr pFile, const SCSVOptions& options = SCSVOptions());

			// Opens the file for reading and loads it.
			static ICSVFilePtr Open(const std::string& fileNameUtf8, const SCSVOptions& options = SCSVOptions());

			const std::vector<std::string>& GetColumns() const override;
			size_t GetRowCount() const override;
			size_t GetColumnIndex(const std::string& column) const override;
			const std::string& GetCell(size_t nRow, size_t nColumn) const override;

		private:
			void Parse(const std::string& text);
			void AddRow(std::vector<std::string>& row, size_t nLine);

		private:
			SCSVOptions m_options;
			std::vector<std::string> m_columns;
			std::vector<std::string> m_cells;	// row by row, m_columns.size() cells in a row
			size_t m_nRows;
			bool m_bFirstRow;
		};
	}
}
