#include <ergodiclib/file_utils.hpp>


std::vector<std::vector<std::vector<double> > > readDemonstrations(const std::string& demonstration_folder_path, int n_dimension) 
{
    std::vector<std::vector<std::vector<double> > > demonstrations;

    std::vector<std::vector<double> > demonstration_vec;
    std::vector<std::string> demonstration_list;  
    std::string filename;
    for (const auto& dirEntry : std::filesystem::recursive_directory_iterator(demonstration_folder_path)) {
        filename = dirEntry.path().generic_string();
        demonstration_list.push_back(filename);
    }

    std::sort(demonstration_list.begin(), demonstration_list.end(), compareStrings);
    for (const std::string& filename : demonstration_list) {
        std::cout << filename << std::endl;
        demonstration_vec = readDemonstrationCSV(filename, n_dimension);
        demonstrations.push_back(demonstration_vec); 
    }

    return demonstrations; 
}

/*
std::vector<std::vector<std::vector<double>>> readDemonstrationsFileList(const std::string& demonstration_file_list_path, int n_dimension) 
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
*/

std::vector<std::vector<double> > readDemonstrationCSV(const std::string& csv_filepath, int n_dimension) 
{
    std::vector<std::vector<double> > demonstration;
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

bool compareStrings(const std::string& file_a, const std::string& file_b) {
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