cmake_minimum_required(VERSION 2.8.12)
project(env)
set (CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED)
pybind11_add_module(env main.cpp card.hpp card.cpp game.cpp game.hpp)
# add_executable(main main.cpp game.cpp game.hpp)
