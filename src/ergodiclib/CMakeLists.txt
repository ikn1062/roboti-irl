cmake_minimum_required(VERSION 3.9)
project(ergodiclib)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -pthread)
endif()

add_compile_definitions(THREADS=0)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(Armadillo REQUIRED)

# Add .cpp files to the library
add_library(${PROJECT_NAME} src/ergodic_measure.cpp src/file_utils.cpp src/fourier_basis.cpp src/num_utils.cpp)

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  ${ARMADILLO_INCLUDE_DIRS})

target_compile_features(${PROJECT_NAME}  PUBLIC cxx_std_20)
target_compile_options(${PROJECT_NAME}  PUBLIC -Wall -Wextra -Wpedantic)

# # Lines below are used to add cpp scripts as ROS2 executable files
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets)
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}-config.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}-targets HAS_LIBRARY_TARGET)

install(PROGRAMS
  scripts/visualize_ergodic.py
  DESTINATION lib/${PROJECT_NAME}
)

# Cross-Compiling options
if(NOT CMAKE_CROSSCOMPILING)
  find_package(Doxygen)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md) # Use the readme in your doxygen docs
  doxygen_add_docs(doxygen include/ src/ README.md ALL)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()