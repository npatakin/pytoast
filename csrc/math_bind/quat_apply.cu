#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>



DEFINE_KERNEL(launch_quat_apply_fwd, {
    const auto [out_ptr, q_ptr, p_ptr] = ptrs;
    out_ptr[offsets[0]] = q_ptr[offsets[1]].apply(p_ptr[offsets[2]]);
})

torch::Tensor quat_apply_fwd(const torch::Tensor& q, const torch::Tensor& p) {
    auto out = torch::empty_like(p);
    auto iter = make_tensor_iterator({out}, {q, p});
    check_operand_last_dims(iter, {3, 4, 3});
    DISPATCH_KERNEL(launch_quat_apply_fwd, vec3*, const quat_t*, const vec3*)
    return out;
}

DEFINE_KERNEL(launch_quat_apply_bwd, {
    const auto [out_dq_ptr, out_dp_ptr, q_ptr, p_ptr, grad_ptr] = ptrs;
    const auto [dq, dp] = CLS(q_ptr)::apply_backward(q_ptr[offsets[2]], p_ptr[offsets[3]], grad_ptr[offsets[4]]);
    out_dq_ptr[offsets[0]] = dq;
    out_dp_ptr[offsets[1]] = dp;
})

auto quat_apply_bwd(const torch::Tensor& q, const torch::Tensor& p, const torch::Tensor& grad) {
    auto grad_q = torch::empty_like(q);
    auto grad_p = torch::empty_like(p);
    auto iter = make_tensor_iterator({grad_q, grad_p}, {q, p, grad});
    check_operand_last_dims(iter, {4, 3, 4, 3, 3});
    DISPATCH_KERNEL(launch_quat_apply_bwd, vec4*, vec3*, const quat_t*, const vec3*, const vec3*)
    return std::tuple{grad_q, grad_p};
}


DEFINE_KERNEL(launch_quat_apply_inv_fwd, {
    const auto [out_ptr, q_ptr, p_ptr] = ptrs;
    out_ptr[offsets[0]] = q_ptr[offsets[1]].apply_inv(p_ptr[offsets[2]]);
})

torch::Tensor quat_apply_inv_fwd(const torch::Tensor& q, const torch::Tensor& p) {
    auto out = torch::empty_like(p);
    auto iter = make_tensor_iterator({out}, {q, p});
    check_operand_last_dims(iter, {3, 4, 3});
    DISPATCH_KERNEL(launch_quat_apply_inv_fwd, vec3*, const quat_t*, const vec3*)
    return out;
}

DEFINE_KERNEL(launch_quat_apply_inv_bwd, {
    const auto [out_dq_ptr, out_dp_ptr, q_ptr, p_ptr, grad_ptr] = ptrs;
    const auto [dq, dp] = CLS(q_ptr)::apply_inv_backward(q_ptr[offsets[2]], p_ptr[offsets[3]], grad_ptr[offsets[4]]);
    out_dq_ptr[offsets[0]] = dq;
    out_dp_ptr[offsets[1]] = dp;
})

auto quat_apply_inv_bwd(const torch::Tensor& q, const torch::Tensor& p, const torch::Tensor& grad) {
    auto grad_q = torch::empty_like(q);
    auto grad_p = torch::empty_like(p);
    auto iter = make_tensor_iterator({grad_q, grad_p}, {q, p, grad});
    check_operand_last_dims(iter, {4, 3, 4, 3, 3});
    DISPATCH_KERNEL(launch_quat_apply_inv_bwd, vec4*, vec3*, const quat_t*, const vec3*, const vec3*)
    return std::tuple{grad_q, grad_p};
}


void bind_quat_apply(py::module_& m) {
    m.def("quat_apply_fwd", &quat_apply_fwd);
    m.def("quat_apply_bwd", &quat_apply_bwd);
    m.def("quat_apply_inv_fwd", &quat_apply_inv_fwd);
    m.def("quat_apply_inv_bwd", &quat_apply_inv_bwd);
}
