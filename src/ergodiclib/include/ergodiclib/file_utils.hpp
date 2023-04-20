#ifndef FILE_UTIL_INCLUDE_GUARD_HPP
#define FILE_UTIL_INCLUDE_GUARD_HPP
/// \file
/// \brief

#include <iosfwd>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <regex>

/// \brief
/// \param demonstration_file_list_path
/// \param n_dimension
/// \return
std::vector<std::vector<std::vector<double>>> readDemonstrations(
  const std::string & demonstration_file_list_path, int n_dimension);

/*
/// \brief
/// \param demonstration_file_list_path
/// \param n_dimension
/// \return
std::vector< std::vector< std::vector<double> > > readDemonstrationsFileList(const std::string& demonstration_file_list_path, int n_dimension);
*/

/// \brief
/// \param csv_filepath
/// \param n_dimension
/// \return
std::vector<std::vector<double>> readDemonstrationCSV(
  const std::string & csv_filepath,
  int n_dimension);

bool compareStrings(const std::string & file_a, const std::string & file_b);

#endif
