#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_quat_mul_fwd, {
    const auto [out_ptr, q1_ptr, q2_ptr] = ptrs;
    out_ptr[offsets[0]] = q1_ptr[offsets[1]].quat_mul(q2_ptr[offsets[2]]);
})

auto quat_mul_fwd(const torch::Tensor& q1, const torch::Tensor& q2) {
    auto out = torch::empty_like(q1);
    auto iter = make_tensor_iterator({out}, {q1, q2});
    check_operand_last_dims(iter, {4, 4, 4});
    DISPATCH_KERNEL(launch_quat_mul_fwd, quat_t*, const quat_t*, const quat_t*)
    return out;
}

DEFINE_KERNEL(launch_quat_mul_bwd, {
    const auto [out_dq1, out_dq2, q1, q2, grad_out] = ptrs;
    CLS(q1)::quat_mul_backward(
        q1[offsets[2]], q2[offsets[3]], grad_out[offsets[4]],
        out_dq1[offsets[0]],
        out_dq2[offsets[1]]
    );
})

auto quat_mul_bwd(const torch::Tensor& q1, const torch::Tensor& q2, const torch::Tensor& grad) {
    auto dq1 = torch::empty_like(q1);
    auto dq2 = torch::empty_like(q2);
    auto iter = make_tensor_iterator({dq1, dq2}, {q1, q2, grad});
    check_operand_last_dims(iter, {4, 4, 4, 4, 4});
    DISPATCH_KERNEL(launch_quat_mul_bwd, vec4*, vec4*, const quat_t*, const quat_t*, const vec4*)
    return std::tuple{dq1, dq2};
}


void bind_quat_mul(py::module_& m) {
    m.def("quat_mul_fwd", &quat_mul_fwd, py::arg("q1"), py::arg("q2"));
    m.def("quat_mul_bwd", &quat_mul_bwd, py::arg("q1"), py::arg("q2"), py::arg("grad_out"));
}
