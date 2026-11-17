#include "CSVFile.h"

#include "CommonLib/exception/exc_base.h"

namespace ML
{
	namespace data
	{
		namespace
		{
			const size_t c_nReadBlockSize = 1024 * 1024;

			std::string ReadAll(CommonLib::file::IFile& file)
			{
				const uint64_t nFileSize = file.GetFileSize();
				if (nFileSize > std::string().max_size())
					throw CommonLib::CExcBase("CSV: the file is too big, size {0}", nFileSize);

				std::string text(size_t(nFileSize), '\0');

				file.SetFileBegin();

				size_t nPos = 0;
				while (nPos < text.size())
				{
					const size_t nBlock = (std::min)(c_nReadBlockSize, text.size() - nPos);
					const std::streamsize nRead = file.Read((byte_t*)&text[nPos], nBlock);
					if (nRead <= 0)
						throw CommonLib::CExcBase("CSV: read {0} bytes of {1}", nPos, text.size());

					nPos += size_t(nRead);
				}

				return text;
			}

			// The text without spaces and tabs at both ends.
			std::string Trim(const std::string& text)
			{
				size_t nBegin = 0;
				size_t nEnd = text.size();
				while (nBegin < nEnd && (text[nBegin] == ' ' || text[nBegin] == '\t'))
					++nBegin;
				while (nEnd > nBegin && (text[nEnd - 1] == ' ' || text[nEnd - 1] == '\t'))
					--nEnd;

				return text.substr(nBegin, nEnd - nBegin);
			}
		}

		CCSVFile::CCSVFile(CommonLib::file::TFilePtr pFile, const SCSVOptions& options) : m_options(options), m_nRows(0), m_bFirstRow(true)
		{
			if (pFile.get() == nullptr)
				throw CommonLib::CExcBase("CSV: the file is null");

			if (m_options.delimiter == '"' || m_options.delimiter == '\r' || m_options.delimiter == '\n')
				throw CommonLib::CExcBase("CSV: the delimiter cannot be a quote or a line break");

			Parse(ReadAll(*pFile));
		}

		ICSVFilePtr CCSVFile::Open(const std::string& fileNameUtf8, const SCSVOptions& options)
		{
			try
			{
				CommonLib::file::TFilePtr pFile = CommonLib::file::CFileCreator::OpenFileA(fileNameUtf8.c_str(),
					CommonLib::file::ofmOpenExisting, CommonLib::file::arRead, CommonLib::file::smRead, CommonLib::file::oftBinary);

				return std::make_shared<CCSVFile>(pFile, options);
			}
			catch (CommonLib::CExcBase& exc)
			{
				exc.AddMsgT("CSV: failed to load the file {0}", fileNameUtf8);
				throw;
			}
		}

		const std::vector<std::string>& CCSVFile::GetColumns() const
		{
			return m_columns;
		}

		size_t CCSVFile::GetRowCount() const
		{
			return m_nRows;
		}

		size_t CCSVFile::GetColumnIndex(const std::string& column) const
		{
			for (size_t i = 0; i < m_columns.size(); ++i)
			{
				if (m_columns[i] == column)
					return i;
			}

			throw CommonLib::CExcBase("CSV: there is no column '{0}'", column);
		}

		const std::string& CCSVFile::GetCell(size_t nRow, size_t nColumn) const
		{
			if (nRow >= m_nRows || nColumn >= m_columns.size())
			{
				throw CommonLib::CExcBase("CSV: cell ({0}, {1}) is out of range, the file has {2} rows and {3} columns",
					nRow, nColumn, m_nRows, m_columns.size());
			}

			return m_cells[nRow * m_columns.size() + nColumn];
		}

		void CCSVFile::Parse(const std::string& text)
		{
			std::vector<std::string> row;	// the fields of the current line, without the last one
			std::string field;				// the field being read
			bool bInQuotes = false;			// between an opening and a closing quote
			bool bQuotedField = false;		// the current field started with a quote
			size_t nLine = 1;				// line number in the file, for the error messages
			size_t nRowLine = 1;			// the line the current row started at

			const auto endRow = [&]()
			{
				// A line with no characters at all is skipped; "" on its own line is a row.
				if (!row.empty() || !field.empty() || bQuotedField)
				{
					row.push_back(field);
					AddRow(row, nRowLine);
				}

				row.clear();
				field.clear();
				bQuotedField = false;
			};

			size_t i = 0;
			if (text.compare(0, 3, "\xEF\xBB\xBF") == 0)
				i = 3;	// UTF-8 byte order mark

			for (; i < text.size(); ++i)
			{
				const char ch = text[i];

				if (bInQuotes)
				{
					if (ch != '"')
					{
						if (ch == '\n')
							++nLine;

						field += ch;
					}
					else if (i + 1 < text.size() && text[i + 1] == '"')
					{
						field += '"';	// "" inside quotes is one quote
						++i;
					}
					else
						bInQuotes = false;
				}
				else if (ch == '"' && field.empty() && !bQuotedField)
				{
					bInQuotes = true;
					bQuotedField = true;
				}
				else if (ch == m_options.delimiter)
				{
					row.push_back(field);
					field.clear();
					bQuotedField = false;
				}
				else if (ch == '\n' || ch == '\r')
				{
					if (ch == '\r' && i + 1 < text.size() && text[i + 1] == '\n')
						++i;	// CRLF is one line break

					endRow();
					++nLine;
					nRowLine = nLine;
				}
				else
					field += ch;
			}

			if (bInQuotes)
				throw CommonLib::CExcBase("CSV: the quote opened in the row at line {0} is not closed", nRowLine);

			endRow();	// the last line when the file does not end with a line break
		}

		void CCSVFile::AddRow(std::vector<std::string>& row, size_t nLine)
		{
			if (m_bFirstRow)
			{
				m_bFirstRow = false;

				if (m_options.hasHeader)
				{
					for (const std::string& name : row)
						m_columns.push_back(Trim(name));

					return;
				}

				for (size_t i = 0; i < row.size(); ++i)
					m_columns.push_back(std::to_string(i));
			}

			if (row.size() != m_columns.size())
				throw CommonLib::CExcBase("CSV: line {0} has {1} fields, expected {2}", nLine, row.size(), m_columns.size());

			for (std::string& cell : row)
				m_cells.push_back(std::move(cell));

			++m_nRows;
		}
	}
}
