#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_std_fwd, {
    const auto [outs, quats] = ptrs;
    outs[offsets[0]] = quats[offsets[1]].std();
})

auto quat_std_fwd(const torch::Tensor& q) {
    auto out = torch::empty_like(q);
    auto iter = make_tensor_iterator({out}, {q});
    check_operand_last_dims(iter, {4, 4});
    DISPATCH_KERNEL(launch_std_fwd, quat_t*, const quat_t*)
    return out;
}

DEFINE_KERNEL(launch_std_bwd, {
    const auto [outs, quats, grads] = ptrs;
    outs[offsets[0]] = CLS(quats)::std_backward(quats[offsets[1]], grads[offsets[2]]);
})

auto quat_std_bwd(const torch::Tensor& q, const torch::Tensor& grad) {
    auto out = torch::empty_like(q);
    auto iter = make_tensor_iterator({out}, {q, grad});
    check_operand_last_dims(iter, {4, 4, 4});
    DISPATCH_KERNEL(launch_std_bwd, vec4*, const quat_t*, const vec4*)
    return out;
}


void bind_quat_std(pybind11::module& m) {
    m.def("quat_std_fwd", &quat_std_fwd, py::arg("quat"));
    m.def("quat_std_bwd", &quat_std_bwd, py::arg("quat"), py::arg("grad"));
}