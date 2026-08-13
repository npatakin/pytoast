#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_quat_from_matrix_fwd, {
    const auto [out, mtx] = ptrs;
    out[offsets[0]] = CLS(out)::from_matrix(mtx + 3 * offsets[1]);
})

torch::Tensor quat_from_matrix_fwd(const torch::Tensor& mtx) {
    auto out_sizes = mtx.sizes().vec();
    out_sizes.pop_back(); out_sizes.back() = 4;
    auto out = torch::empty(out_sizes, mtx.options());

    auto flat_sizes = mtx.sizes().vec();
    flat_sizes.pop_back(); flat_sizes.back() = 9;
    auto mtx_flat = mtx.reshape(flat_sizes);

    auto iter = make_tensor_iterator({out}, {mtx_flat});
    check_operand_last_dims(iter, {4, 9});
    DISPATCH_KERNEL(launch_quat_from_matrix_fwd, quat_t*, const vec3*)
    return out;
}


DEFINE_KERNEL(launch_quat_from_matrix_bwd, {
    const auto [d_mtx, mtx, grad] = ptrs;
    CLS(grad)::from_matrix_backward(mtx + 3 * offsets[1], grad[offsets[2]].q, d_mtx + 3 * offsets[0]);
})

torch::Tensor quat_from_matrix_bwd(const torch::Tensor& mtx, const torch::Tensor& grad_output) {
    auto flat_sizes = mtx.sizes().vec();
    flat_sizes.pop_back(); flat_sizes.back() = 9;

    auto mtx_flat = mtx.reshape(flat_sizes);
    auto d_mtx_flat = torch::empty(flat_sizes, mtx.options());

    auto iter = make_tensor_iterator({d_mtx_flat}, {mtx_flat, grad_output});
    check_operand_last_dims(iter, {9, 9, 4});
    DISPATCH_KERNEL(launch_quat_from_matrix_bwd, vec3*, const vec3*, const quat_t*)
    return d_mtx_flat.view(mtx.sizes());
}


void bind_quat_from_matrix(py::module_& m) {
    m.def("quat_from_matrix_fwd", &quat_from_matrix_fwd, py::arg("mtx"));
    m.def("quat_from_matrix_bwd", &quat_from_matrix_bwd, py::arg("mtx"), py::arg("grad_output"));
}
