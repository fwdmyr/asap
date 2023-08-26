# Works with 3.14 and tested through 3.26
cmake_minimum_required(VERSION 3.14...3.26)

project(
  asap
  VERSION 0.1
  DESCRIPTION "Algorithms for Sparse Assignment Problems"
  LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  option(ASAP_ENABLE_TESTS "Build tests" ON)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(HEADERS_LIST
    include/sparse_matrix.hpp
  )

set(TARGET_NAME asap)

add_library(${TARGET_NAME} INTERFACE)
target_sources(${TARGET_NAME} INTERFACE HEADERS_LIST)
target_include_directories(${TARGET_NAME} INTERFACE include)
target_compile_features(${TARGET_NAME} INTERFACE cxx_std_17)
target_compile_options(${TARGET_NAME} INTERFACE -Wall -Wextra -Wpedantic -Werror)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND ASAP_ENABLE_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
