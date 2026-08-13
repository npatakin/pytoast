#ifndef PYTOAST_INTERPOLATION_CUH
#define PYTOAST_INTERPOLATION_CUH

#include <torch/extension.h>

void bind_interpolation(py::module_& m);

torch::Tensor extrapolate_indices(const torch::Tensor& masks, uint extrapolate_window);

#endif //PYTOAST_INTERPOLATION_CUH
