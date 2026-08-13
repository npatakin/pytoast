#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_inverse_fwd, {
    const auto [outs, quats] = ptrs;
    outs[offsets[0]] = quats[offsets[1]].inverse();
})

auto quat_inverse_fwd(const torch::Tensor& q) {
    auto out = torch::empty_like(q);
    auto iter = make_tensor_iterator({out}, {q});
    check_operand_last_dims(iter, {4, 4});
    DISPATCH_KERNEL(launch_inverse_fwd, quat_t*, const quat_t*)
    return out;
}

DEFINE_KERNEL(launch_inverse_bwd, {
    const auto [outs, quats, grads] = ptrs;
    outs[offsets[0]] = CLS(quats)::inverse_backward(quats[offsets[1]], grads[offsets[2]]);
})

auto quat_inverse_bwd(const torch::Tensor& q, const torch::Tensor& grad) {
    auto out = torch::empty_like(q);
    auto iter = make_tensor_iterator({out}, {q, grad});
    check_operand_last_dims(iter, {4, 4, 4});
    DISPATCH_KERNEL(launch_inverse_bwd, vec4*, const quat_t*, const vec4*)
    return out;
}


DEFINE_KERNEL(launch_conjugate_fwd, {
    const auto [outs, quats] = ptrs;
    outs[offsets[0]] = quats[offsets[1]].conjugate();
})

auto quat_conjugate_fwd(const torch::Tensor& q) {
    auto out = torch::empty_like(q);
    auto iter = make_tensor_iterator({out}, {q});
    check_operand_last_dims(iter, {4, 4});
    DISPATCH_KERNEL(launch_conjugate_fwd, quat_t*, const quat_t*)
    return out;
}

DEFINE_KERNEL(launch_conjugate_bwd, {
    const auto [outs, grads] = ptrs;
    outs[offsets[0]] = CLS(grads)::conjugate_backward(grads[offsets[2]].q);
})

auto quat_conjugate_bwd(const torch::Tensor& grad) {
    auto out = torch::empty_like(grad);
    auto iter = make_tensor_iterator({out}, {grad});
    check_operand_last_dims(iter, {4, 4});
    DISPATCH_KERNEL(launch_conjugate_bwd, vec4*, const quat_t*)
    return out;
}


void bind_quat_inverse(pybind11::module& m) {
    m.def("quat_inverse_fwd", &quat_inverse_fwd, py::arg("quat"));
    m.def("quat_inverse_bwd", &quat_inverse_bwd, py::arg("quat"), py::arg("grad"));

    m.def("quat_conjugate_fwd", &quat_conjugate_fwd, py::arg("q"));
    m.def("quat_conjugate_bwd", &quat_conjugate_bwd, py::arg("grad"));
}
