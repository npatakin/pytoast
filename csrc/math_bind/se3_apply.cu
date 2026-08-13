#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_se3_apply_fwd, {
    const auto [out, q, t, p] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    out[offsets[0]] = se3_type(q[offsets[1]], t[offsets[2]]).apply(p[offsets[3]]);
})

auto se3_apply_fwd(
    const torch::Tensor& q, const torch::Tensor& t, const torch::Tensor& p
) {
    auto out = torch::empty_like(p);
    auto iter = make_tensor_iterator({out}, {q, t, p});
    check_operand_last_dims(iter, {3, 4, 3, 3});
    DISPATCH_KERNEL(launch_se3_apply_fwd, vec3*, const quat_t*, const vec3*, const vec3*)
    return out;
}


DEFINE_KERNEL(launch_se3_apply_bwd, {
    const auto [dq, dt, dp, q, t, p, grad] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    const auto [dse3, dt_val] = se3_type::apply_backward(
        se3_type(q[offsets[3]], t[offsets[4]]),
        p[offsets[5]], grad[offsets[6]]
    );
    dq[offsets[0]] = std::get<0>(dse3);
    dt[offsets[1]] = dt_val;
    dp[offsets[2]] = std::get<1>(dse3);
})

auto se3_apply_bwd(
    const torch::Tensor& q, const torch::Tensor& t,
    const torch::Tensor& p, const torch::Tensor& grad
) {
    auto dq = torch::empty_like(q);
    auto dt = torch::empty_like(t);
    auto dp = torch::empty_like(p);
    auto iter = make_tensor_iterator({dq, dt, dp}, {q, t, p, grad});
    check_operand_last_dims(iter, {4, 3, 3, 4, 3, 3, 3});
    DISPATCH_KERNEL(launch_se3_apply_bwd,
        vec4*, vec3*, vec3*,
        const quat_t*, const vec3*, const vec3*, const vec3*
    )
    return std::tuple{dq, dt, dp};
}


DEFINE_KERNEL(launch_se3_apply_inv_fwd, {
    const auto [out, q, t, p] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    out[offsets[0]] = se3_type(q[offsets[1]], t[offsets[2]]).apply_inv(p[offsets[3]]);
})

auto se3_apply_inv_fwd(
    const torch::Tensor& q, const torch::Tensor& t, const torch::Tensor& p
) {
    auto out = torch::empty_like(p);
    auto iter = make_tensor_iterator({out}, {q, t, p});
    check_operand_last_dims(iter, {3, 4, 3, 3});
    DISPATCH_KERNEL(launch_se3_apply_inv_fwd, vec3*, const quat_t*, const vec3*, const vec3*)
    return out;
}


DEFINE_KERNEL(launch_se3_apply_inv_bwd, {
    const auto [dq, dt, dp, q, t, p, grad] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    const auto [dse3, dt_val] = se3_type::apply_inv_backward(
        se3_type(q[offsets[3]], t[offsets[4]]),
        p[offsets[5]], grad[offsets[6]]
    );
    dq[offsets[0]] = std::get<0>(dse3);
    dt[offsets[1]] = dt_val;
    dp[offsets[2]] = std::get<1>(dse3);
})

auto se3_apply_inv_bwd(
    const torch::Tensor& q, const torch::Tensor& t,
    const torch::Tensor& p, const torch::Tensor& grad
) {
    auto dq = torch::empty_like(q);
    auto dt = torch::empty_like(t);
    auto dp = torch::empty_like(p);
    auto iter = make_tensor_iterator({dq, dt, dp}, {q, t, p, grad});
    check_operand_last_dims(iter, {4, 3, 3, 4, 3, 3, 3});
    DISPATCH_KERNEL(launch_se3_apply_inv_bwd,
        vec4*, vec3*, vec3*,
        const quat_t*, const vec3*, const vec3*, const vec3*
    )
    return std::tuple{dq, dt, dp};
}


void bind_se3_apply(py::module_& m) {
    m.def("se3_apply_fwd", &se3_apply_fwd, py::arg("q"), py::arg("t"), py::arg("p"));
    m.def("se3_apply_bwd", &se3_apply_bwd,
        py::arg("q"), py::arg("t"), py::arg("p"), py::arg("grad_out"));
    m.def("se3_apply_inv_fwd", &se3_apply_inv_fwd, py::arg("q"), py::arg("t"), py::arg("p"));
    m.def("se3_apply_inv_bwd", &se3_apply_inv_bwd,
        py::arg("q"), py::arg("t"), py::arg("p"), py::arg("grad_out"));
}
