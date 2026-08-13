#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>



DEFINE_KERNEL(launch_slerp_fwd, {
    const auto [out_q, q1, q2, t] = ptrs;
    out_q[offsets[0]] = q1[offsets[1]].slerp(q2[offsets[2]], t[offsets[3]]);
})

torch::Tensor quat_slerp_fwd(const torch::Tensor& q1, const torch::Tensor& q2, const torch::Tensor& t) {
    auto out = torch::empty_like(q1);
    auto iter = make_tensor_iterator({out}, {q1, q2, t});
    check_operand_last_dims(iter, {4, 4, 4, 1});
    DISPATCH_KERNEL(launch_slerp_fwd, quat_t*, const quat_t*, const quat_t*, const scalar_t*)
    return out;
}

DEFINE_KERNEL(launch_slerp_bwd, {
    const auto [out_dq1, out_dq2, out_dt, q1, q2, t, grad_q] = ptrs;
    const auto [dq1, dq2, dt] = CLS(q1)::slerp_backward(
        q1[offsets[3]].q, q2[offsets[4]].q, t[offsets[5]], grad_q[offsets[6]]
    );
    out_dq1[offsets[0]] = dq1;
    out_dq2[offsets[1]] = dq2;
    out_dt[offsets[2]] = dt;
})

auto quat_slerp_bwd(
    const torch::Tensor& q1,
    const torch::Tensor& q2,
    const torch::Tensor& t,
    const torch::Tensor& grad_q
) {
    auto dq1 = torch::empty_like(q1);
    auto dq2 = torch::empty_like(q2);
    auto dt = torch::empty_like(t);
    auto iter = make_tensor_iterator({dq1, dq2, dt}, {q1, q2, t, grad_q});
    check_operand_last_dims(iter, {4, 4, 1, 4, 4, 1, 4});

    DISPATCH_KERNEL_THREADS(launch_slerp_bwd, CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS/2,
        vec4*, vec4*, scalar_t*,
        const quat_t*, const quat_t*, const scalar_t*,
        const vec4*
    )

    return std::tuple{dq1, dq2, dt};
}


void bind_quat_slerp(py::module &m) {
    m.def("quat_slerp_fwd", &quat_slerp_fwd, py::arg("q1"), py::arg("q2"), py::arg("t"));
    m.def("quat_slerp_bwd", &quat_slerp_bwd, py::arg("q1"), py::arg("q2"), py::arg("t"), py::arg("grad_output"));
}


