#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_to_axis_angle_fwd, {
    const auto [out, q] = ptrs;
    out[offsets[0]] = q[offsets[1]].to_axis_angle();
})

torch::Tensor quat_to_axis_angle_fwd(const torch::Tensor& q) {
    auto sizes = q.sizes().vec();
    sizes.back() = 3;
    auto out = torch::empty(sizes, q.options());
    auto iter = make_tensor_iterator({out}, {q});
    check_operand_last_dims(iter, {3, 4});
    DISPATCH_KERNEL(launch_to_axis_angle_fwd, vec3*, const quat_t*)
    return out;
}

DEFINE_KERNEL(launch_to_axis_angle_bwd, {
    const auto [out, q, grad] = ptrs;
    out[offsets[0]] = CLS(q)::to_axis_angle_backward(q[offsets[1]], grad[offsets[2]]);
})

torch::Tensor quat_to_axis_angle_bwd(const torch::Tensor& q, const torch::Tensor& grad_output) {
    auto out = torch::empty_like(q);
    auto iter = make_tensor_iterator({out}, {q, grad_output});
    check_operand_last_dims(iter, {4, 4, 3});
    DISPATCH_KERNEL(launch_to_axis_angle_bwd, vec4*, const quat_t*, const vec3*)
    return out;
}


void bind_quat_to_axis_angle(py::module_& m) {
    m.def("quat_to_axis_angle_fwd", &quat_to_axis_angle_fwd, py::arg("quat"));
    m.def("quat_to_axis_angle_bwd", &quat_to_axis_angle_bwd, py::arg("quat"), py::arg("grad_output"));
}
