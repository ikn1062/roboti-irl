#ifndef FILE_UTIL_INCLUDE_GUARD_HPP
#define FILE_UTIL_INCLUDE_GUARD_HPP
/// \file
/// \brief File utility functions for getting demonstrations from CSVs

#include <iosfwd>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <regex>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif


namespace ergodiclib
{
/// \brief Reads Demonstrations from a given directory path
/// \param demonstration_dir_path Directory path for demonstrations
/// \param n_dimension Number of dimensions in dynamic system
/// \return Vector of demonstration trajectories
std::vector<arma::mat> readDemonstrations(
  const std::string & demonstration_dir_path,
  const int n_dimension);

/// \brief Read Demonstrations from a given CSV file
/// \param csv_filepath File Path for demonstration csv
/// \param n_dimension Number of dimensions in dynamic system
/// \return Matrix of demonstration trajectory
arma::mat readDemonstrationCSV(const std::string & csv_filepath, const int n_dimension);

/// \brief Save trajectory and controls into a csv file
/// \param filename Name of file to be saved
/// \param trajectory Trajectory and controls to be saved to file
/// \return None
void saveTrajectoryCSV(
  const std::string & filename, const std::pair<arma::mat,
  arma::mat> & trajectory);
}

#endif
