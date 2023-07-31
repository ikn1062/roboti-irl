#include <ergodiclib/file_utils.hpp>


static bool compareStrings(const std::string & file_a, const std::string & file_b)
{
  // Define regular expression to match numeric part of string
  std::regex num_regex("\\d+");

  // Extract numeric part of each string
  std::smatch a_match, b_match;
  std::regex_search(file_a, a_match, num_regex);
  std::regex_search(file_b, b_match, num_regex);
  int a_num = std::stoi(a_match[0]);
  int b_num = std::stoi(b_match[0]);

  // Compare the numeric parts
  return a_num < b_num;
}


namespace ergodiclib
{
std::vector<arma::mat> readDemonstrations(
  const std::string & demonstration_dir_path,
  const int n_dimension)
{
  std::vector<arma::mat> demonstrations;

  // Get list of all files in the folder
  arma::mat demonstration_vec;
  std::vector<std::string> demonstration_list;
  std::string filename;
  for (const auto & dirEntry :
    std::filesystem::recursive_directory_iterator(demonstration_dir_path))
  {
    filename = dirEntry.path().generic_string();
    if (filename.substr(filename.find_last_of(".") + 1) == "csv") {
      demonstration_list.push_back(filename);
    }
  }

  // Sort the folder by number and collect demonstrations
  std::sort(demonstration_list.begin(), demonstration_list.end(), compareStrings);
  for (const std::string & filename : demonstration_list) {
    std::cout << "Reading: " << filename << std::endl;
    demonstration_vec = readDemonstrationCSV(filename, n_dimension);
    demonstrations.push_back(demonstration_vec);
  }

  return demonstrations;
}

arma::mat readDemonstrationCSV(const std::string & csv_filepath, const int n_dimension)
{
  arma::mat demonstration(0, 0, arma::fill::zeros);
  std::ifstream demonstration_file(csv_filepath);

  std::vector<double> line_values;
  arma::vec col_vec;
  std::string line, value;
  int demo_len;
  double number;
  while (std::getline(demonstration_file, line)) {
    std::stringstream ss(line);
    line_values.clear();

    for (int i = 0; i < n_dimension; i++) {
      std::getline(ss, value, ',');
      number = std::stod(value);
      line_values.push_back(number);
    }
    col_vec = arma::conv_to<arma::vec>::from(line_values);
    demo_len = demonstration.n_cols;
    demonstration.resize(col_vec.n_elem, demo_len);
    demonstration.insert_cols(demo_len, col_vec);
  }

  demonstration_file.close();
  return demonstration;
}

void saveTrajectoryCSV(
  const std::string & filename, const std::pair<arma::mat,
  arma::mat> & trajectory)
{
  std::string x_file = filename + "_trajectory.csv";
  std::string u_file = filename + "_control.csv";

  arma::mat XT = trajectory.first.t();
  arma::mat UT = trajectory.second.t();

  XT.save(x_file, arma::csv_ascii);
  UT.save(u_file, arma::csv_ascii);

  return;
}
}
