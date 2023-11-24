# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(solver)

# 设置源文件目录
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp)

# 使用 file 命令获取源文件列表
file(GLOB STATIC_LIBRARY_SOURCE_FILES
    ${SOURCE_DIR}/*.cpp
)
file(GLOB SHARED_LIBRARY_SOURCE_FILES
    ${SOURCE_DIR}/*.cpp
)

file(GLOB EXECUTABLE_SOURCE_FILES
    ${SOURCE_DIR}/*.cpp
)
# 生成动态链接库
add_library(solver_shared SHARED ${SHARED_LIBRARY_SOURCE_FILES})

# 生成静态库
add_library(solver_static STATIC ${STATIC_LIBRARY_SOURCE_FILES})

# 生成可执行文件
add_executable(solver_executable ${EXECUTABLE_SOURCE_FILES})

# 可选：设置输出目录
set_target_properties(solver_shared PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set_target_properties(solver_shared PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(solver_static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/lib)
set_target_properties(solver_executable PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/bin)

