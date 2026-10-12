#ifndef PYTOAST_INTERPOLATION_CUH
#define PYTOAST_INTERPOLATION_CUH

#include <torch/csrc/utils/pybind.h>

void bind_interpolation(py::module_& m);

at::Tensor extrapolate_indices(const at::Tensor& masks, uint extrapolate_window);

#endif //PYTOAST_INTERPOLATION_CUH
