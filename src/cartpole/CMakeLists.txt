cmake_minimum_required(VERSION 3.8)
project(cartpole)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ergodiclib REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Catch2 3 REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

file(WRITE src/null.cpp "")
add_library(${PROJECT_NAME}
    src/null.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  ${ARMADILLO_INCLUDE_DIRS})

target_compile_features(${PROJECT_NAME}  PUBLIC cxx_std_20)
target_compile_options(${PROJECT_NAME}  PUBLIC -Wall -Wextra -Wpedantic)

add_executable(cartpole_key src/cartpole_key.cpp)
ament_target_dependencies(cartpole_key rclcpp std_msgs geometry_msgs sensor_msgs Armadillo)
target_link_libraries(cartpole_key ${rclcpp_LIBRARIES})
target_link_libraries(cartpole_key ${ARMADILLO_LIBRARIES})

add_executable(cartpole_ctrl src/cartpole_control.cpp)
ament_target_dependencies(cartpole_ctrl rclcpp std_msgs geometry_msgs sensor_msgs std_srvs Armadillo)
target_link_libraries(cartpole_ctrl ${rclcpp_LIBRARIES} ${PROJECT_NAME})
target_link_libraries(cartpole_ctrl ${ARMADILLO_LIBRARIES})
target_link_libraries(cartpole_ctrl ergodiclib::ergodiclib "${cpp_typesupport_target}")

add_executable(ergodic_script src/ergodic_script.cpp)
ament_target_dependencies(ergodic_script Armadillo)
target_link_libraries(ergodic_script ${ARMADILLO_LIBRARIES} ${PROJECT_NAME})
target_link_libraries(ergodic_script ergodiclib::ergodiclib "${cpp_typesupport_target}")

add_executable(controller_script src/controller_script.cpp)
ament_target_dependencies(controller_script Armadillo)
target_link_libraries(controller_script ${ARMADILLO_LIBRARIES} ${PROJECT_NAME})
target_link_libraries(controller_script ergodiclib::ergodiclib "${cpp_typesupport_target}")


install(TARGETS
  cartpole_key
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS cartpole_ctrl ergodic_script controller_script ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets)
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}-config.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}-targets HAS_LIBRARY_TARGET)

install(PROGRAMS
  scripts/cartpole_file_util.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch src urdf
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)

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

include(CTest)
add_executable(test_cartpole tests/test_cartpole.cpp)
target_link_libraries(test_cartpole Catch2::Catch2WithMain ergodiclib::ergodiclib ${PROJECT_NAME} ${ARMADILLO_LIBRARIES})
add_test(NAME Test_of_Cartpole_Model COMMAND test_cartpole)

add_executable(test_ilqr_controller tests/test_ilqr_controller.cpp)
target_link_libraries(test_ilqr_controller Catch2::Catch2WithMain ergodiclib::ergodiclib ${PROJECT_NAME} ${ARMADILLO_LIBRARIES})
add_test(NAME Test_of_iLQR_Control COMMAND test_ilqr_controller)

add_executable(test_ergodic tests/test_ergodic.cpp)
target_link_libraries(test_ergodic Catch2::Catch2WithMain ergodiclib::ergodiclib ${PROJECT_NAME} ${ARMADILLO_LIBRARIES})
add_test(NAME Test_of_Ergoidic_Measure COMMAND test_ergodic)

ament_package()
