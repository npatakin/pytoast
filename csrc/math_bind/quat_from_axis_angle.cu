#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_from_axis_angle_fwd, {
    const auto [out, aa] = ptrs;
    out[offsets[0]] = CLS(out)::from_axis_angle(aa[offsets[1]]);
})

torch::Tensor quat_from_axis_angle_fwd(const torch::Tensor& axis_angle) {
    auto sizes = axis_angle.sizes().vec();
    sizes.back() = 4;
    auto out = torch::empty(sizes, axis_angle.options());
    auto iter = make_tensor_iterator({out}, {axis_angle});
    check_operand_last_dims(iter, {4, 3});
    DISPATCH_KERNEL(launch_from_axis_angle_fwd, quat_t*, const vec3*)
    return out;
}

DEFINE_KERNEL(launch_from_axis_angle_bwd, {
    const auto [out, aa, grad] = ptrs;
    out[offsets[0]] = CLS(grad)::from_axis_angle_backward(aa[offsets[1]], grad[offsets[2]].q);
})

torch::Tensor quat_from_axis_angle_bwd(const torch::Tensor& axis_angle, const torch::Tensor& grad_output) {
    auto out = torch::empty_like(axis_angle);
    auto iter = make_tensor_iterator({out}, {axis_angle, grad_output});
    check_operand_last_dims(iter, {3, 3, 4});
    DISPATCH_KERNEL(launch_from_axis_angle_bwd, vec3*, const vec3*, const quat_t*)
    return out;
}


void bind_quat_from_axis_angle(py::module_& m) {
    m.def("quat_from_axis_angle_fwd", &quat_from_axis_angle_fwd, py::arg("axis_angle"));
    m.def("quat_from_axis_angle_bwd", &quat_from_axis_angle_bwd, py::arg("axis_angle"), py::arg("grad_output"));
}
