cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)

include_directories("d:\\Apps\\opencv\\build\\include")
link_directories("d:\\Apps\\opencv\\build\\bin")

add_executable(filter src/filter.cu)
 