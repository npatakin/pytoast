#include "quat_bind.h"
#include <toast/math/se3.h>
#include <toast/utils/launch_helpers.h>


DEFINE_KERNEL(launch_random_quat, {
    const auto [out, r] = ptrs;
    out[offsets[0]] = CLS(out)::random_from_uniform(r[offsets[1]]);
})

torch::Tensor random_quat_from_uniform(const torch::Tensor& rand_vals) {
    auto sizes = rand_vals.sizes().vec();
    sizes.back() = 4;
    auto out = torch::empty(sizes, rand_vals.options());
    auto iter = make_tensor_iterator({out}, {rand_vals});
    check_operand_last_dims(iter, {4, 3});
    DISPATCH_KERNEL(launch_random_quat, quat_t*, const vec3*)
    return out;
}


DEFINE_KERNEL(launch_quat_from_xyzw_fwd, {
    const auto [out, in] = ptrs;
    const auto in_vec = in[offsets[1]];
    out[offsets[0]] = CLS(out)::from_xyzw(in_vec.x, in_vec.y, in_vec.z, in_vec.w);
})

torch::Tensor quat_from_xyzw_fwd(const torch::Tensor& xyzw) {
    auto out = torch::empty_like(xyzw);
    auto iter = make_tensor_iterator({out}, {xyzw});
    check_operand_last_dims(iter, {4, 4});
    DISPATCH_KERNEL(launch_quat_from_xyzw_fwd, quat_t*, const vec4*)
    return out;
}


DEFINE_KERNEL(launch_quat_from_xyzw_bwd, {
    const auto [out, grad] = ptrs;
    out[offsets[0]] = CLS(grad)::from_xyzw_backward(grad[offsets[1]].q);
})

torch::Tensor quat_from_xyzw_bwd(const torch::Tensor& grad_output) {
    auto out = torch::empty_like(grad_output);
    auto iter = make_tensor_iterator({out}, {grad_output});
    check_operand_last_dims(iter, {4, 4});
    DISPATCH_KERNEL(launch_quat_from_xyzw_bwd, vec4*, const quat_t*)
    return out;
}

DEFINE_KERNEL(launch_quat_unit, {
    const auto [out_quat] = ptrs;
    out_quat[offsets[0]] = CLS(out_quat)();
})

torch::Tensor quat_unit(const torch::Tensor& out) {
    auto iter = make_tensor_iterator({out}, {}, {-1}, false);
    DISPATCH_KERNEL(launch_quat_unit, quat_t*)
    return out;
}


void bind_quat_creation(py::module_& m) {
    m.def("random_quat_from_uniform", &random_quat_from_uniform, py::arg("rand_vals"));
    m.def("quat_from_xyzw_fwd", &quat_from_xyzw_fwd, py::arg("xyzw"));
    m.def("quat_from_xyzw_bwd", &quat_from_xyzw_bwd, py::arg("grad_output"));
    m.def("quat_unit_", &quat_unit, py::arg("quat"));
}
