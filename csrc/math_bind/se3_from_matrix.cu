#include "se3_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_se3_from_matrix_3x4_fwd, {
    const auto [out_q, out_t, mtx] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(out_q)::T>>;
    const auto result = se3_type::from_matrix(mtx + 3 * offsets[2]);
    out_q[offsets[0]] = result.q;
    out_t[offsets[1]] = result.t;
})

DEFINE_KERNEL(launch_se3_from_matrix_4x4_fwd, {
    const auto [out_q, out_t, mtx] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(out_q)::T>>;
    const auto result = se3_type::from_matrix(mtx + 4 * offsets[2]);
    out_q[offsets[0]] = result.q;
    out_t[offsets[1]] = result.t;
})

auto se3_from_matrix_fwd(const torch::Tensor& mtx) {
    const int64_t num_rows = mtx.size(-2);
    if (num_rows != 3 && num_rows != 4) {
        throw std::runtime_error("se3_from_matrix_fwd: mtx must be [..., 3, 4] or [..., 4, 4]");
    }

    auto batch_sizes = mtx.sizes().vec();
    batch_sizes.pop_back(); batch_sizes.pop_back();

    auto q_sizes = batch_sizes; q_sizes.push_back(4);
    auto t_sizes = batch_sizes; t_sizes.push_back(3);
    auto out_q = torch::empty(q_sizes, mtx.options());
    auto out_t = torch::empty(t_sizes, mtx.options());

    auto flat_sizes = batch_sizes;
    flat_sizes.push_back(num_rows * 4);
    auto mtx_flat = mtx.reshape(flat_sizes);

    auto iter = make_tensor_iterator({out_q, out_t}, {mtx_flat});
    check_operand_last_dims(iter, {4, 3, num_rows * 4});
    if (num_rows == 3) {
        DISPATCH_KERNEL(launch_se3_from_matrix_3x4_fwd, quat_t*, vec3*, const vec4*)
    } else {
        DISPATCH_KERNEL(launch_se3_from_matrix_4x4_fwd, quat_t*, vec3*, const vec4*)
    }
    return std::tuple{out_q, out_t};
}


DEFINE_KERNEL(launch_se3_from_matrix_3x4_bwd, {
    const auto [grad_mtx, mtx, grad_q, grad_t] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(grad_q)::T>>;
    se3_type::from_matrix_backward(
        mtx + 3 * offsets[1],
        grad_q[offsets[2]].q,
        grad_t[offsets[3]],
        grad_mtx + 3 * offsets[0]
    );
})

DEFINE_KERNEL(launch_se3_from_matrix_4x4_bwd, {
    const auto [grad_mtx, mtx, grad_q, grad_t] = ptrs;
    using se3_type = se3<dtype_t<typename CLS(grad_q)::T>>;
    se3_type::from_matrix_backward(
        mtx + 4 * offsets[1],
        grad_q[offsets[2]].q,
        grad_t[offsets[3]],
        grad_mtx + 4 * offsets[0]
    );
})

torch::Tensor se3_from_matrix_bwd(
    const torch::Tensor& mtx,
    const torch::Tensor& grad_q, const torch::Tensor& grad_t
) {
    const int64_t num_rows = mtx.size(-2);

    auto flat_sizes = mtx.sizes().vec();
    flat_sizes.pop_back(); flat_sizes.back() = num_rows * 4;
    auto mtx_flat = mtx.reshape(flat_sizes);
    auto grad_mtx_flat = torch::zeros(flat_sizes, mtx.options());

    auto iter = make_tensor_iterator({grad_mtx_flat}, {mtx_flat, grad_q, grad_t});
    check_operand_last_dims(iter, {num_rows * 4, num_rows * 4, 4, 3});
    if (num_rows == 3) {
        DISPATCH_KERNEL(launch_se3_from_matrix_3x4_bwd, vec4*, const vec4*, const quat_t*, const vec3*)
    } else {
        DISPATCH_KERNEL(launch_se3_from_matrix_4x4_bwd, vec4*, const vec4*, const quat_t*, const vec3*)
    }
    return grad_mtx_flat.view(mtx.sizes());
}


void bind_se3_from_matrix(py::module_& m) {
    m.def("se3_from_matrix_fwd", &se3_from_matrix_fwd, py::arg("mtx"));
    m.def("se3_from_matrix_bwd", &se3_from_matrix_bwd,
        py::arg("mtx"), py::arg("grad_q"), py::arg("grad_t"));
}
