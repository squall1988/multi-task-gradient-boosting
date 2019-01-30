cmake_minimum_required(VERSION 3.6)
project(mtree)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

set(SOURCE_FILES main.cpp tree.h Node.cpp Node.h Dataset.cpp Dataset.h utils.h Random.h ModelSelection.h ModelSelection.cpp error.h Updater.h Loss.h SingleTaskUpdater.cpp SingleTaskUpdater.h MultiTaskUpdater.h MultiTaskUpdater.cpp tree.cpp LinearLoss.cpp LinearLoss.h Booster.cpp Booster.h ThreadPool.h)
add_executable(mtree ${SOURCE_FILES})