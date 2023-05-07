cmake_minimum_required(VERSION 3.9)
project(ergodiclib)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(Armadillo REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_library(${PROJECT_NAME} src/ergodic_measure.cpp src/num_utils.cpp src/file_utils.cpp src/controller.cpp src/fourier_basis.cpp)

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  ${ARMADILLO_INCLUDE_DIRS})

target_compile_features(${PROJECT_NAME}  PUBLIC cxx_std_17)
target_compile_options(${PROJECT_NAME}  PUBLIC -Wall -Wextra -Wpedantic)

add_executable(ergodic_test src/ergodic_test.cpp)
ament_target_dependencies(ergodic_test Armadillo)
target_link_libraries(ergodic_test ${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

add_executable(ergodic_main src/ergodic_main.cpp)
ament_target_dependencies(ergodic_main Armadillo)
target_link_libraries(ergodic_main ${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

install(TARGETS ergodic_test ergodic_main DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)
install(TARGETS ergodic_test ergodic_main ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets)
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}-config.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}-targets HAS_LIBRARY_TARGET)

#if (NOT CMAKE_CROSSCOMPILING)
#  find_package(Doxygen)
#  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md) # Use the readme in your doxygen docs
#  doxygen_add_docs(doxygen include/ src/ README.md ALL)
#endif()

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

# include(CTest)
# add_executable(ergodiclib_test tests/tests.cpp)
# target_link_libraries(ergodiclib_test ${PROJECT_NAME} Catch2::Catch2WithMain)
# add_test(NAME Test_of_Ergodic_lib COMMAND ergodiclib_test)

ament_package()