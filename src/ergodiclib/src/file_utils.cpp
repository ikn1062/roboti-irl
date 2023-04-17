#include <ergodiclib/file_utils.hpp>


std::vector<std::vector<std::vector<double>>> readDemonstrations(const std::string& demonstration_file_list_path, int n_dimension) 
{
    std::vector<std::vector<std::vector<double>>> demonstrations;
    std::ifstream demonstrationListFile(demonstration_file_list_path);

    std::string filename;
    std::vector<std::vector<double>> demonstration_vec;
    while (std::getline(demonstrationListFile, filename)) {
        demonstration_vec = readDemonstrationCSV(filename, n_dimension);
        demonstrations.push_back(demonstration_vec);
    }

    demonstrationListFile.close();
    return demonstrations; 
}


std::vector<std::vector<double>> readDemonstrationCSV(const std::string& csv_filepath, int n_dimension) 
{
    std::vector<std::vector<double>> demonstration;
    std::ifstream demonstration_file(csv_filepath);

    std::string line;
    while (std::getline(demonstration_file, line)) {
        std::stringstream ss(line);
        std::vector<double> line_values;

        for (int i = 0; i < n_dimension; i++) {
            std::string value;
            std::getline(ss, value, ',');
            double number = std::stod(value);
            line_values.push_back(number);
        }

        demonstration.push_back(line_values);
  }

  demonstration_file.close();
  return demonstration;
}