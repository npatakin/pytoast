#ifndef PYTOAST_SE3_BIND_H
#define PYTOAST_SE3_BIND_H

#include <torch/extension.h>
#include <toast/utils/tensor_helpers.h>

void bind_se3_apply(py::module& m);
void bind_se3_inverse(py::module& m);
void bind_se3_compose(py::module& m);
void bind_se3_slerp(py::module& m);
void bind_se3_to_matrix(py::module& m);
void bind_se3_from_matrix(py::module& m);

void bind_se3(py::module& m);

#endif //PYTOAST_SE3_BIND_H
