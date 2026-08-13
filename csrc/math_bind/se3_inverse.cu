#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_se3_inverse_fwd, {
    const auto [out_q, out_t, q, t] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    const auto result = se3_type(q[offsets[2]], t[offsets[3]]).inverse();
    out_q[offsets[0]] = result.q;
    out_t[offsets[1]] = result.t;
})

auto se3_inverse_fwd(const torch::Tensor& q, const torch::Tensor& t) {
    auto out_q = torch::empty_like(q);
    auto out_t = torch::empty_like(t);
    auto iter = make_tensor_iterator({out_q, out_t}, {q, t});
    check_operand_last_dims(iter, {4, 3, 4, 3});
    DISPATCH_KERNEL(launch_se3_inverse_fwd, quat_t*, vec3*, const quat_t*, const vec3*)
    return std::tuple{out_q, out_t};
}


DEFINE_KERNEL(launch_se3_inverse_bwd, {
    const auto [out_q, out_t, q, t, grad_q, grad_t] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    const auto [dq, dt] = se3_type::inverse_backward(
        se3_type(q[offsets[2]], t[offsets[3]]),
        grad_q[offsets[4]], grad_t[offsets[5]]
    );
    out_q[offsets[0]] = dq;
    out_t[offsets[1]] = dt;
})

auto se3_inverse_bwd(
    const torch::Tensor& q, const torch::Tensor& t,
    const torch::Tensor& grad_q, const torch::Tensor& grad_t
) {
    auto out_q = torch::empty_like(q);
    auto out_t = torch::empty_like(t);
    auto iter = make_tensor_iterator({out_q, out_t}, {q, t, grad_q, grad_t});
    check_operand_last_dims(iter, {4, 3, 4, 3, 4, 3});
    DISPATCH_KERNEL(launch_se3_inverse_bwd,
        vec4*, vec3*,
        const quat_t*, const vec3*, const vec4*, const vec3*
    )
    return std::tuple{out_q, out_t};
}


void bind_se3_inverse(py::module_& m) {
    m.def("se3_inverse_fwd", &se3_inverse_fwd, py::arg("q"), py::arg("t"));
    m.def("se3_inverse_bwd", &se3_inverse_bwd,
        py::arg("q"), py::arg("t"), py::arg("grad_q_out"), py::arg("grad_t_out"));
}
