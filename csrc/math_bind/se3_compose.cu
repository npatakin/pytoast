#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_se3_compose_fwd, {
    const auto [out_q, out_t, q1, t1, q2, t2] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q1)::T>>;
    const auto result = se3_type(q1[offsets[2]], t1[offsets[3]]).compose(
        se3_type(q2[offsets[4]], t2[offsets[5]])
    );
    out_q[offsets[0]] = result.q;
    out_t[offsets[1]] = result.t;
})

auto se3_compose_fwd(
    const torch::Tensor& q1, const torch::Tensor& t1,
    const torch::Tensor& q2, const torch::Tensor& t2
) {
    auto out_q = torch::empty_like(q1);
    auto out_t = torch::empty_like(t1);
    auto iter = make_tensor_iterator({out_q, out_t}, {q1, t1, q2, t2});
    check_operand_last_dims(iter, {4, 3, 4, 3, 4, 3});
    DISPATCH_KERNEL(launch_se3_compose_fwd, quat_t*, vec3*, const quat_t*, const vec3*, const quat_t*, const vec3*)
    return std::tuple{out_q, out_t};
}


DEFINE_KERNEL(launch_se3_compose_bwd, {
    const auto [dq1, dt1, dq2, dt2, q1, t1, q2, t2, grad_q, grad_t] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q1)::T>>;
    const auto [dq1_val, dt1_val, dq2_val, dt2_val] = se3_type::compose_backward(
        se3_type(q1[offsets[4]], t1[offsets[5]]),
        se3_type(q2[offsets[6]], t2[offsets[7]]),
        grad_q[offsets[8]], grad_t[offsets[9]]
    );
    dq1[offsets[0]] = dq1_val;
    dt1[offsets[1]] = dt1_val;
    dq2[offsets[2]] = dq2_val;
    dt2[offsets[3]] = dt2_val;
})

auto se3_compose_bwd(
    const torch::Tensor& q1, const torch::Tensor& t1,
    const torch::Tensor& q2, const torch::Tensor& t2,
    const torch::Tensor& grad_q, const torch::Tensor& grad_t
) {
    auto dq1 = torch::empty_like(q1);
    auto dt1 = torch::empty_like(t1);
    auto dq2 = torch::empty_like(q2);
    auto dt2 = torch::empty_like(t2);
    auto iter = make_tensor_iterator({dq1, dt1, dq2, dt2}, {q1, t1, q2, t2, grad_q, grad_t});
    check_operand_last_dims(iter, {4, 3, 4, 3, 4, 3, 4, 3, 4, 3});
    DISPATCH_KERNEL(launch_se3_compose_bwd,
        vec4*, vec3*, vec4*, vec3*,
        const quat_t*, const vec3*, const quat_t*, const vec3*,
        const vec4*, const vec3*
    )
    return std::tuple{dq1, dt1, dq2, dt2};
}


void bind_se3_compose(py::module_& m) {
    m.def("se3_compose_fwd", &se3_compose_fwd,
        py::arg("q1"), py::arg("t1"), py::arg("q2"), py::arg("t2"));
    m.def("se3_compose_bwd", &se3_compose_bwd,
        py::arg("q1"), py::arg("t1"), py::arg("q2"), py::arg("t2"),
        py::arg("grad_quat_out"), py::arg("grad_t_out"));
}
