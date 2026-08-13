#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_se3_slerp_fwd, {
    const auto [out_q, out_t, q1, t1, q2, t2, weight] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q1)::T>>;
    const auto result = se3_type(q1[offsets[2]], t1[offsets[3]]).slerp(
        se3_type(q2[offsets[4]], t2[offsets[5]]), weight[offsets[6]]
    );
    out_q[offsets[0]] = result.q;
    out_t[offsets[1]] = result.t;
})

auto se3_slerp_fwd(
    const torch::Tensor& q1, const torch::Tensor& t1,
    const torch::Tensor& q2, const torch::Tensor& t2,
    const torch::Tensor& weight
) {
    auto out_q = torch::empty_like(q1);
    auto out_t = torch::empty_like(t1);
    auto iter = make_tensor_iterator({out_q, out_t}, {q1, t1, q2, t2, weight});
    check_operand_last_dims(iter, {4, 3, 4, 3, 4, 3, 1});
    DISPATCH_KERNEL(launch_se3_slerp_fwd,
        quat_t*, vec3*,
        const quat_t*, const vec3*, const quat_t*, const vec3*, const scalar_t*
    )
    return std::tuple{out_q, out_t};
}


DEFINE_KERNEL(launch_se3_slerp_bwd, {
    const auto [dq1, dt1, dq2, dt2, dw, q1, t1, q2, t2, weight, grad_q, grad_t] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q1)::T>>;
    using quat_type = typename se3_type::quat_t;
    const auto [dq1_val, dt1_val, dq2_val, dt2_val, dw_val] = se3_type::slerp_backward(
        se3_type(q1[offsets[5]], t1[offsets[6]]),
        se3_type(q2[offsets[7]], t2[offsets[8]]),
        weight[offsets[9]],
        se3_type(quat_type(grad_q[offsets[10]]), grad_t[offsets[11]])
    );
    dq1[offsets[0]] = dq1_val;
    dt1[offsets[1]] = dt1_val;
    dq2[offsets[2]] = dq2_val;
    dt2[offsets[3]] = dt2_val;
    dw[offsets[4]] = dw_val;
})

auto se3_slerp_bwd(
    const torch::Tensor& q1, const torch::Tensor& t1,
    const torch::Tensor& q2, const torch::Tensor& t2,
    const torch::Tensor& weight,
    const torch::Tensor& grad_q, const torch::Tensor& grad_t
) {
    auto dq1 = torch::empty_like(q1);
    auto dt1 = torch::empty_like(t1);
    auto dq2 = torch::empty_like(q2);
    auto dt2 = torch::empty_like(t2);
    auto dw = torch::empty_like(weight);
    auto iter = make_tensor_iterator({dq1, dt1, dq2, dt2, dw}, {q1, t1, q2, t2, weight, grad_q, grad_t});
    check_operand_last_dims(iter, {4, 3, 4, 3, 1, 4, 3, 4, 3, 1, 4, 3});
    DISPATCH_KERNEL_THREADS(launch_se3_slerp_bwd, CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS/2,
        vec4*, vec3*, vec4*, vec3*, scalar_t*,
        const quat_t*, const vec3*, const quat_t*, const vec3*, const scalar_t*,
        const vec4*, const vec3*
    )
    return std::tuple{dq1, dt1, dq2, dt2, dw};
}


void bind_se3_slerp(py::module_& m) {
    m.def("se3_slerp_fwd", &se3_slerp_fwd,
        py::arg("q1"), py::arg("t1"), py::arg("q2"), py::arg("t2"), py::arg("weight"));
    m.def("se3_slerp_bwd", &se3_slerp_bwd,
        py::arg("q1"), py::arg("t1"), py::arg("q2"), py::arg("t2"), py::arg("weight"),
        py::arg("grad_quat_out"), py::arg("grad_t_out"));
}
