#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_from_euler_fwd, {
    const auto [out, angles] = ptrs;
    out[offsets[0]] = CLS(out)::from_euler_angles(angles[offsets[1]]);
})

torch::Tensor quat_from_euler_angles_fwd(const torch::Tensor& euler_angles) {
    auto sizes = euler_angles.sizes().vec();
    sizes.back() = 4;
    auto out = torch::empty(sizes, euler_angles.options());
    auto iter = make_tensor_iterator({out}, {euler_angles});
    check_operand_last_dims(iter, {4, 3});
    DISPATCH_KERNEL(launch_from_euler_fwd, quat_t*, const vec3*)
    return out;
}

DEFINE_KERNEL(launch_from_euler_bwd, {
    const auto [out, angles, grad] = ptrs;
    out[offsets[0]] = CLS(grad)::from_euler_angles_bwd(angles[offsets[1]], grad[offsets[2]].q);
})

torch::Tensor quat_from_euler_angles_bwd(const torch::Tensor& euler_angles, const torch::Tensor& grad_output) {
    auto out = torch::empty_like(euler_angles);
    auto iter = make_tensor_iterator({out}, {euler_angles, grad_output});
    check_operand_last_dims(iter, {3, 3, 4});
    DISPATCH_KERNEL(launch_from_euler_bwd, vec3*, const vec3*, const quat_t*)
    return out;
}


void bind_quat_from_euler_angles(py::module_& m) {
    m.def("quat_from_euler_angles_fwd", &quat_from_euler_angles_fwd, py::arg("euler_angles"));
    m.def("quat_from_euler_angles_bwd", &quat_from_euler_angles_bwd, py::arg("euler_angles"), py::arg("grad_output"));
}
