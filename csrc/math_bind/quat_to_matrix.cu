#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_to_matrix_fwd, {
    const auto [out, q] = ptrs;
    q[offsets[1]].to_matrix(out + 3 * offsets[0]);
})

at::Tensor quat_to_matrix_fwd(const at::Tensor& q) {
    auto sizes = q.sizes().vec();
    sizes.back() = 9;
    auto out = at::empty(sizes, q.options());
    auto iter = make_tensor_iterator({out}, {q});
    check_operand_last_dims(iter, {9, 4});
    DISPATCH_KERNEL(launch_to_matrix_fwd, vec3*, const quat_t*)

    auto result_sizes = q.sizes().vec();
    result_sizes.back() = 3;
    result_sizes.push_back(3);
    return out.view(result_sizes);
}

DEFINE_KERNEL(launch_to_matrix_bwd, {
    const auto [out, q, grad] = ptrs;
    out[offsets[0]] = CLS(q)::to_matrix_backward(q[offsets[1]], grad + 3 * offsets[2]);
})

at::Tensor quat_to_matrix_bwd(const at::Tensor& q, const at::Tensor& grad_output) {
    auto out = at::empty_like(q);

    auto grad_sizes = q.sizes().vec();
    grad_sizes.back() = 9;
    auto grad_flat = grad_output.reshape(grad_sizes);

    auto iter = make_tensor_iterator({out}, {q, grad_flat});
    check_operand_last_dims(iter, {4, 4, 9});
    DISPATCH_KERNEL(launch_to_matrix_bwd, vec4*, const quat_t*, const vec3*)
    return out;
}


void bind_quat_to_matrix(py::module_& m) {
    m.def("quat_to_matrix_fwd", &quat_to_matrix_fwd, py::arg("quat"));
    m.def("quat_to_matrix_bwd", &quat_to_matrix_bwd, py::arg("quat"), py::arg("grad_output"));
}
