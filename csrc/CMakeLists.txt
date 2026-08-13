cmake_minimum_required(VERSION 3.31)
project(pytoast CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Development)
add_library(${PROJECT_NAME} STATIC
        ext.cpp

        ../src/toast/include/toast/math/quat.h
        ../src/toast/include/toast/math/se3.h
        ../src/toast/include/toast/math/scalar.h
        ../src/toast/include/toast/utils/launch_helpers.h
        ../src/toast/include/toast/math/motion.h

        math_bind/quat_bind.h
        math_bind/se3_bind.h
        math_bind/quat_bind.cpp
        math_bind/se3_bind.cpp
        math_bind/quat_creation.cu
        math_bind/quat_from_axis_angle.cu
        math_bind/quat_to_axis_angle.cu
        math_bind/quat_to_matrix.cu
        math_bind/quat_from_matrix.cu
        math_bind/quat_inverse.cu
        math_bind/quat_std.cu
        math_bind/quat_mul.cu
        math_bind/quat_apply.cu
        math_bind/quat_slerp.cu
        math_bind/se3_apply.cu
        math_bind/se3_compose.cu
        math_bind/se3_inverse.cu
        math_bind/se3_slerp.cu
        math_bind/se3_from_matrix.cu
        math_bind/se3_to_matrix.cu

        motion_bind/velocities.cu
        motion_bind/velocities.cuh
        motion_bind/interpolation.cu
        motion_bind/interpolation.cuh
        motion_bind/extrapolation_indices.h
)
target_compile_options(
        ${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

execute_process(
        COMMAND
            "python"
            -c "import torch.utils.cpp_extension;print(';'.join(torch.utils.cpp_extension.include_paths(device_type='cuda')))"
        OUTPUT_VARIABLE PYTORCH_INCLUDES
        OUTPUT_STRIP_TRAILING_WHITESPACE)

target_include_directories(
        ${PROJECT_NAME} PRIVATE
        ${PYTORCH_INCLUDES}
        ${Python3_INCLUDE_DIRS}
        ../src/toast/include
)
