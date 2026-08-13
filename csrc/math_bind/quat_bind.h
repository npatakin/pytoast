#ifndef PYTOAST_QUAT_BIND_H
#define PYTOAST_QUAT_BIND_H

#include <torch/extension.h>

#include <toast/utils/tensor_helpers.h>

void bind_quat_creation(py::module_& m);
void bind_quat_from_axis_angle(py::module_& m);
void bind_quat_to_axis_angle(py::module_& m);
void bind_quat_to_matrix(py::module_& m);
void bind_quat_from_matrix(py::module_& m);
void bind_quat_inverse(py::module_& m);
void bind_quat_std(py::module_& m);
void bind_quat_mul(py::module_& m);
void bind_quat_apply(py::module_& m);
void bind_quat_slerp(py::module_& m);
void bind_quat_from_euler_angles(py::module_& m);

void bind_quat(py::module_& m);

#endif //PYTOAST_QUAT_BIND_H
