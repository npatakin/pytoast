#include "se3_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_se3_to_matrix_3x4_fwd, {
    const auto [out, q, t] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    se3_type(q[offsets[1]], t[offsets[2]]).to_matrix_3x4(out + 3 * offsets[0]);
})

torch::Tensor se3_to_matrix_3x4_fwd(const torch::Tensor& q, const torch::Tensor& t) {
    auto sizes = q.sizes().vec();
    sizes.back() = 12;
    auto out = torch::empty(sizes, q.options());
    auto iter = make_tensor_iterator({out}, {q, t});
    check_operand_last_dims(iter, {12, 4, 3});
    DISPATCH_KERNEL(launch_se3_to_matrix_3x4_fwd, vec4*, const quat_t*, const vec3*)

    auto result_sizes = q.sizes().vec();
    result_sizes.back() = 3;
    result_sizes.push_back(4);
    return out.view(result_sizes);
}


DEFINE_KERNEL(launch_se3_to_matrix_3x4_bwd, {
    const auto [dq, dt, q, t, grad] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    const auto [dq_val, dt_val] = se3_type::to_matrix_3x4_backward(
        se3_type(q[offsets[2]], t[offsets[3]]),
        grad + 3 * offsets[4]
    );
    dq[offsets[0]] = dq_val;
    dt[offsets[1]] = dt_val;
})

auto se3_to_matrix_3x4_bwd(
    const torch::Tensor& q, const torch::Tensor& t,
    const torch::Tensor& grad_mtx
) {
    auto dq = torch::empty_like(q);
    auto dt = torch::empty_like(t);

    auto grad_sizes = q.sizes().vec();
    grad_sizes.back() = 12;
    auto grad_flat = grad_mtx.reshape(grad_sizes);

    auto iter = make_tensor_iterator({dq, dt}, {q, t, grad_flat});
    check_operand_last_dims(iter, {4, 3, 4, 3, 12});
    DISPATCH_KERNEL(launch_se3_to_matrix_3x4_bwd, vec4*, vec3*, const quat_t*, const vec3*, const vec4*)
    return std::tuple{dq, dt};
}


DEFINE_KERNEL(launch_se3_to_matrix_4x4_fwd, {
    const auto [out, q, t] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    se3_type(q[offsets[1]], t[offsets[2]]).to_matrix_4x4(out + 4 * offsets[0]);
})

torch::Tensor se3_to_matrix_4x4_fwd(const torch::Tensor& q, const torch::Tensor& t) {
    auto sizes = q.sizes().vec();
    sizes.back() = 16;
    auto out = torch::empty(sizes, q.options());
    auto iter = make_tensor_iterator({out}, {q, t});
    check_operand_last_dims(iter, {16, 4, 3});
    DISPATCH_KERNEL(launch_se3_to_matrix_4x4_fwd, vec4*, const quat_t*, const vec3*)

    auto result_sizes = q.sizes().vec();
    result_sizes.push_back(4);
    return out.view(result_sizes);
}


DEFINE_KERNEL(launch_se3_to_matrix_4x4_bwd, {
    const auto [dq, dt, q, t, grad] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(q)::T>>;
    const auto [dq_val, dt_val] = se3_type::to_matrix_4x4_backward(
        se3_type(q[offsets[2]], t[offsets[3]]),
        grad + 4 * offsets[4]
    );
    dq[offsets[0]] = dq_val;
    dt[offsets[1]] = dt_val;
})

auto se3_to_matrix_4x4_bwd(
    const torch::Tensor& q, const torch::Tensor& t,
    const torch::Tensor& grad_mtx
) {
    auto dq = torch::empty_like(q);
    auto dt = torch::empty_like(t);

    auto grad_sizes = q.sizes().vec();
    grad_sizes.back() = 16;
    auto grad_flat = grad_mtx.reshape(grad_sizes);

    auto iter = make_tensor_iterator({dq, dt}, {q, t, grad_flat});
    check_operand_last_dims(iter, {4, 3, 4, 3, 16});
    DISPATCH_KERNEL(launch_se3_to_matrix_4x4_bwd, vec4*, vec3*, const quat_t*, const vec3*, const vec4*)
    return std::tuple{dq, dt};
}


void bind_se3_to_matrix(py::module_& m) {
    m.def("se3_to_matrix_3x4_fwd", &se3_to_matrix_3x4_fwd, py::arg("q"), py::arg("t"));
    m.def("se3_to_matrix_3x4_bwd", &se3_to_matrix_3x4_bwd,
        py::arg("q"), py::arg("t"), py::arg("grad_mtx_out"));
    m.def("se3_to_matrix_4x4_fwd", &se3_to_matrix_4x4_fwd, py::arg("q"), py::arg("t"));
    m.def("se3_to_matrix_4x4_bwd", &se3_to_matrix_4x4_bwd,
        py::arg("q"), py::arg("t"), py::arg("grad_mtx_out"));
}
