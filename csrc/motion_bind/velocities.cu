#include "velocities.cuh"

#include <toast/math/motion.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>


DEFINE_KERNEL(launch_velocities_kernel_fwd, {
    const auto [lin_ptr, ang_ptr, q1_ptr, t1_ptr, q2_ptr, t2_ptr, time1_ptr, time2_ptr] = ptrs;

    const auto [linear_v, angular_v] = estimate_velocities(
        q1_ptr[offsets[2]], t1_ptr[offsets[3]],
        q2_ptr[offsets[4]], t2_ptr[offsets[5]],
        time1_ptr[offsets[6]], time2_ptr[offsets[7]]
    );

    lin_ptr[offsets[0]] = linear_v;
    ang_ptr[offsets[1]] = angular_v;
})


auto velocities_fwd(
    const torch::Tensor& q1,
    const torch::Tensor& t1,
    const torch::Tensor& q2,
    const torch::Tensor& t2,
    const torch::Tensor& left_time,
    const torch::Tensor& right_time
) {
    auto out_linear = torch::empty_like(t1);
    auto out_angular = torch::empty_like(t1);
    auto iter = make_tensor_iterator(
        {out_linear, out_angular},
        {q1, t1, q2, t2, left_time, right_time}
    );
    check_operand_last_dims(iter, {3, 3, 4, 3, 4, 3, 1, 1});

    DISPATCH_KERNEL(launch_velocities_kernel_fwd,
        vec3*, vec3*,
        const quat_t*, const vec3*, const quat_t*, const vec3*, const scalar_t*, const scalar_t*
    )

    return std::tuple{out_linear, out_angular};
}


DEFINE_KERNEL(launch_velocities_kernel_bwd, {
    const auto [
        dq1, dt1, dq2, dt2, dleft, dright,
        q1_ptr, t1_ptr, q2_ptr, t2_ptr, time1_ptr, time2_ptr,
        d_linear, d_angular
    ] = ptrs;

    const auto [
        dq1_val, dt1_val, dq2_val, dt2_val, dleft_val, dright_val
    ] = estimate_velocities_backward(
        q1_ptr[offsets[6]], t1_ptr[offsets[7]], q2_ptr[offsets[8]], t2_ptr[offsets[9]],
        time1_ptr[offsets[10]], time2_ptr[offsets[11]],
        d_linear[offsets[12]], d_angular[offsets[13]]
    );

    dq1[offsets[0]] = dq1_val;
    dt1[offsets[1]] = dt1_val;
    dq2[offsets[2]] = dq2_val;
    dt2[offsets[3]] = dt2_val;
    dleft[offsets[4]] = dleft_val;
    dright[offsets[5]] = dright_val;
})

auto velocities_bwd(
    const torch::Tensor& q1,
    const torch::Tensor& t1,
    const torch::Tensor& q2,
    const torch::Tensor& t2,
    const torch::Tensor& left_time,
    const torch::Tensor& right_time,
    const torch::Tensor& d_linear,
    const torch::Tensor& d_angular
) {
     auto dq1 = torch::empty_like(q1);
     auto dt1 = torch::empty_like(t1);
     auto dq2 = torch::empty_like(q2);
     auto dt2 = torch::empty_like(t2);
     auto dleft_time = torch::empty_like(left_time);
     auto dright_time = torch::empty_like(right_time);

    auto iter = make_tensor_iterator(
        {dq1, dt1, dq2, dt2, dleft_time, dright_time},
        {q1, t1, q2, t2, left_time, right_time, d_linear, d_angular}
    );

    DISPATCH_KERNEL_THREADS(
        launch_velocities_kernel_bwd,
        CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS / 2,
        vec4*, vec3*, vec4*, vec3*, scalar_t*, scalar_t*,
        const quat_t*, const vec3*, const quat_t*, const vec3*,
        const scalar_t*, const scalar_t*,
        const vec3*, const vec3*
        )

    return std::tuple{dq1, dt1, dq2, dt2, dleft_time, dright_time};
}



DEFINE_KERNEL(launch_apply_velocity_fwd, {
    const auto [out_quat, out_t, q0_ptr, t0_ptr, v_ptr, w_ptr, delta_t_ptr] = ptrs;

    const auto [q, t] = apply_velocity(
        q0_ptr[offsets[2]], t0_ptr[offsets[3]], v_ptr[offsets[4]], w_ptr[offsets[5]],
        delta_t_ptr[offsets[6]]
    );
    out_quat[offsets[0]] = q;
    out_t[offsets[1]] = t;
})

auto apply_velocity_fwd(
    const torch::Tensor& q0,
    const torch::Tensor& t0,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& delta_t
) {
    auto out_q = torch::empty_like(q0);
    auto out_t = torch::empty_like(t0);

    auto iter = make_tensor_iterator({out_q, out_t}, {q0, t0, v, w, delta_t});
    check_operand_last_dims(iter, {4, 3, 4, 3, 3, 3, 1});

    DISPATCH_KERNEL(launch_apply_velocity_fwd, quat_t*, vec3*,
        const quat_t*, const vec3*, const vec3*, const vec3*, const scalar_t*)

    return std::tuple{out_q, out_t};
}


DEFINE_KERNEL(launch_apply_velocity_bwd, {
    const auto [
        dq0, dt0, dv, dw, d_delta_t,
        q0, t0, v, w, delta_t, grad_q, grad_t
    ] = ptrs;

    const auto [dq0_val, dt0_val, dv_val, dw_val, d_delta_t_val] = apply_velocity_backward(
        q0[offsets[5]], t0[offsets[6]], v[offsets[7]], w[offsets[8]], delta_t[offsets[9]],
        grad_q[offsets[10]], grad_t[offsets[11]]
    );
    dq0[offsets[0]] = dq0_val;
    dt0[offsets[1]] = dt0_val;
    dv[offsets[2]] = dv_val;
    dw[offsets[3]] = dw_val;
    d_delta_t[offsets[4]] = d_delta_t_val;
})

auto apply_velocity_bwd(
    const torch::Tensor& q0,
    const torch::Tensor& t0,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& delta_t,
    const torch::Tensor& grad_q,
    const torch::Tensor& grad_t
) {
    auto dq0 = torch::empty_like(q0);
    auto dt0 = torch::empty_like(t0);
    auto dv = torch::empty_like(v);
    auto dw = torch::empty_like(w);
    auto d_delta_t = torch::empty_like(delta_t);

    auto iter = make_tensor_iterator(
        {dq0, dt0, dv, dw, d_delta_t},
        {q0, t0, v, w, delta_t, grad_q, grad_t}
    );
    check_operand_last_dims(iter, {4, 3, 3, 3, 1, 4, 3, 3, 3, 1, 4, 3});

    DISPATCH_KERNEL_THREADS(launch_apply_velocity_bwd,
        CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS / 2,
        vec4*, vec3*, vec3*, vec3*, scalar_t*,
        const quat_t*, const vec3*, const vec3*, const vec3*, const scalar_t*, const vec4*, const vec3*
        )

    return std::tuple{dq0, dt0, dv, dw, d_delta_t};
}


DEFINE_KERNEL(launch_apply_linear_motion_fwd, {
    const auto [
        out_ptr,
        q0_ptr, t0_ptr, v_ptr, w_ptr, pose_time_ptr,
        point_ptr, point_time_ptr
    ] = ptrs;

    out_ptr[offsets[0]] = transform_point_dynamic<dtype_t<CLS(pose_time_ptr)>>(
        q0_ptr[offsets[1]], t0_ptr[offsets[2]], v_ptr[offsets[3]], w_ptr[offsets[4]],
        pose_time_ptr[offsets[5]], point_ptr[offsets[6]], point_time_ptr[offsets[7]]
    );
})


auto linear_motion_transform_fwd(
    const torch::Tensor& q0,
    const torch::Tensor& t0,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& pose_time,
    const torch::Tensor& points,
    const torch::Tensor& points_time
) {
    auto out = torch::empty_like(points);
    auto iter = make_tensor_iterator(
        {out},
        {q0, t0, v, w, pose_time, points, points_time}
    );
    check_operand_last_dims(iter, {3, 4, 3, 3, 3, 1, 3, 1});

    DISPATCH_KERNEL(launch_apply_linear_motion_fwd,
        vec3*, const quat_t*, const vec3*, const vec3*, const vec3*, const scalar_t*, const vec3*, const scalar_t*)

    return out;
}


DEFINE_KERNEL(launch_apply_linear_motion_bwd, {
    const auto [
        dq0_ptr, dt0_ptr, dv_ptr, dw_ptr, dpose_time_ptr, dpoints_ptr, dpoints_time_ptr,  // 0-6
        q0_ptr,   t0_ptr,  v_ptr,  w_ptr,  pose_time_ptr,  point_ptr,   point_time_ptr,  // 7-13
        grad_ptr // 14
    ] = ptrs;

    transform_point_dynamic_backward(
        q0_ptr[offsets[7]], t0_ptr[offsets[8]], v_ptr[offsets[9]], w_ptr[offsets[10]],
        pose_time_ptr[offsets[11]], point_ptr[offsets[12]], point_time_ptr[offsets[13]],
        grad_ptr[offsets[14]],

        dq0_ptr[offsets[0]], dt0_ptr[offsets[1]], dv_ptr[offsets[2]], dw_ptr[offsets[3]],
        dpose_time_ptr[offsets[4]], dpoints_ptr[offsets[5]], dpoints_time_ptr[offsets[6]]
    );
})


auto linear_motion_transform_bwd(
    const torch::Tensor& q0,
    const torch::Tensor& t0,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& pose_time,
    const torch::Tensor& points,
    const torch::Tensor& points_time,
    const torch::Tensor& grads
) {
    auto dq0 = torch::empty_like(q0);
    auto dt0 = torch::empty_like(t0);
    auto dv = torch::empty_like(v);
    auto dw = torch::empty_like(w);
    auto dpose_time = torch::empty_like(pose_time);
    auto dpoint = torch::empty_like(points);
    auto dpoint_time = torch::empty_like(points_time);

    auto iter = make_tensor_iterator(
        {dq0, dt0, dv, dw, dpose_time, dpoint, dpoint_time},
        {q0, t0, v, w, pose_time, points, points_time, grads}
    );
    DISPATCH_KERNEL_THREADS(launch_apply_linear_motion_bwd,
        CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS / 2,
        vec4*, vec3*, vec3*, vec3*, scalar_t*, vec3*, scalar_t*,
        const quat_t*, const vec3*, const vec3*, const vec3*, const scalar_t*, const vec3*, const scalar_t*, const vec3*
        )

    return std::tuple{dq0, dt0, dv, dw, dpose_time, dpoint, dpoint_time};
}


DEFINE_KERNEL(launch_linear_velocity_fwd, {
    const auto [out_ptr, t1_ptr, t2_ptr, dt_ptr] = ptrs;
    out_ptr[offsets[0]] = linear_velocity<dtype_t<CLS(dt_ptr)>>(
        t1_ptr[offsets[1]], t2_ptr[offsets[2]], dt_ptr[offsets[3]]
    );
})

auto linear_velocity_fwd(
    const torch::Tensor& t1,
    const torch::Tensor& t2,
    const torch::Tensor& dt
) {
    auto out = torch::empty_like(t1);
    auto iter = make_tensor_iterator({out}, {t1, t2, dt});
    check_operand_last_dims(iter, {3, 3, 3, 1});
    DISPATCH_KERNEL(launch_linear_velocity_fwd,
        vec3*, const vec3*, const vec3*, const scalar_t*)
    return out;
}


DEFINE_KERNEL(launch_linear_velocity_bwd, {
    const auto [dt1, dt2, ddt, t1_ptr, t2_ptr, dt_ptr, d_out] = ptrs;
    const auto [dt1_val, dt2_val, ddt_val] = linear_velocity_backward<dtype_t<CLS(dt_ptr)>>(
        t1_ptr[offsets[3]], t2_ptr[offsets[4]], dt_ptr[offsets[5]], d_out[offsets[6]]
    );
    dt1[offsets[0]] = dt1_val;
    dt2[offsets[1]] = dt2_val;
    ddt[offsets[2]] = ddt_val;
})

auto linear_velocity_bwd(
    const torch::Tensor& t1,
    const torch::Tensor& t2,
    const torch::Tensor& dt,
    const torch::Tensor& d_out
) {
    auto dt1 = torch::empty_like(t1);
    auto dt2 = torch::empty_like(t2);
    auto ddt = torch::empty_like(dt);
    auto iter = make_tensor_iterator({dt1, dt2, ddt}, {t1, t2, dt, d_out});
    check_operand_last_dims(iter, {3, 3, 1, 3, 3, 1, 3});
    DISPATCH_KERNEL_THREADS(launch_linear_velocity_bwd,
        CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS / 2,
        vec3*, vec3*, scalar_t*,
        const vec3*, const vec3*, const scalar_t*, const vec3*)
    return std::tuple{dt1, dt2, ddt};
}


DEFINE_KERNEL(launch_angular_velocity_fwd, {
    const auto [out_ptr, q1_ptr, q2_ptr, dt_ptr] = ptrs;
    out_ptr[offsets[0]] = angular_velocity_short_arc(
        q1_ptr[offsets[1]], q2_ptr[offsets[2]], dt_ptr[offsets[3]]
    );
})

auto angular_velocity_fwd(
    const torch::Tensor& q1,
    const torch::Tensor& q2,
    const torch::Tensor& dt
) {
    auto sizes = q1.sizes().vec();
    sizes.back() = 3;
    auto out = torch::empty(sizes, q1.options());
    auto iter = make_tensor_iterator({out}, {q1, q2, dt});
    check_operand_last_dims(iter, {3, 4, 4, 1});
    DISPATCH_KERNEL(launch_angular_velocity_fwd,
        vec3*, const quat_t*, const quat_t*, const scalar_t*)
    return out;
}


DEFINE_KERNEL(launch_angular_velocity_bwd, {
    const auto [dq1, dq2, ddt, q1_ptr, q2_ptr, dt_ptr, d_out] = ptrs;
    const auto [dq1_val, dq2_val, ddt_val] = angular_velocity_short_arc_backward(
        q1_ptr[offsets[3]], q2_ptr[offsets[4]], dt_ptr[offsets[5]], d_out[offsets[6]]
    );
    dq1[offsets[0]] = dq1_val;
    dq2[offsets[1]] = dq2_val;
    ddt[offsets[2]] = ddt_val;
})

auto angular_velocity_bwd(
    const torch::Tensor& q1,
    const torch::Tensor& q2,
    const torch::Tensor& dt,
    const torch::Tensor& d_out
) {
    auto dq1 = torch::empty_like(q1);
    auto dq2 = torch::empty_like(q2);
    auto ddt = torch::empty_like(dt);
    auto iter = make_tensor_iterator({dq1, dq2, ddt}, {q1, q2, dt, d_out});
    check_operand_last_dims(iter, {4, 4, 1, 4, 4, 1, 3});
    DISPATCH_KERNEL_THREADS(launch_angular_velocity_bwd,
        CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS / 2,
        vec4*, vec4*, scalar_t*,
        const quat_t*, const quat_t*, const scalar_t*, const vec3*)
    return std::tuple{dq1, dq2, ddt};
}


DEFINE_KERNEL(launch_apply_angular_velocity_fwd, {
    const auto [out_ptr, q0_ptr, w_ptr, dt_ptr] = ptrs;
    out_ptr[offsets[0]] = apply_angular_velocity(
        q0_ptr[offsets[1]], w_ptr[offsets[2]], dt_ptr[offsets[3]]
    );
})

auto apply_angular_velocity_fwd(
    const torch::Tensor& q0,
    const torch::Tensor& w,
    const torch::Tensor& delta_t
) {
    auto out = torch::empty_like(q0);
    auto iter = make_tensor_iterator({out}, {q0, w, delta_t});
    check_operand_last_dims(iter, {4, 4, 3, 1});
    DISPATCH_KERNEL(launch_apply_angular_velocity_fwd,
        quat_t*, const quat_t*, const vec3*, const scalar_t*)
    return out;
}


DEFINE_KERNEL(launch_apply_angular_velocity_bwd, {
    const auto [dq0, dw, ddt, q0_ptr, w_ptr, dt_ptr, d_quat] = ptrs;
    const auto [dq0_val, dw_val, ddt_val] = apply_angular_velocity_backward(
        q0_ptr[offsets[3]], w_ptr[offsets[4]], dt_ptr[offsets[5]], d_quat[offsets[6]]
    );
    dq0[offsets[0]] = dq0_val;
    dw[offsets[1]] = dw_val;
    ddt[offsets[2]] = ddt_val;
})

auto apply_angular_velocity_bwd(
    const torch::Tensor& q0,
    const torch::Tensor& w,
    const torch::Tensor& delta_t,
    const torch::Tensor& d_quat
) {
    auto dq0 = torch::empty_like(q0);
    auto dw = torch::empty_like(w);
    auto d_delta_t = torch::empty_like(delta_t);
    auto iter = make_tensor_iterator({dq0, dw, d_delta_t}, {q0, w, delta_t, d_quat});
    check_operand_last_dims(iter, {4, 3, 1, 4, 3, 1, 4});
    DISPATCH_KERNEL_THREADS(launch_apply_angular_velocity_bwd,
        CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS / 2,
        vec4*, vec3*, scalar_t*,
        const quat_t*, const vec3*, const scalar_t*, const vec4*)
    return std::tuple{dq0, dw, d_delta_t};
}


void bind_velocities(py::module_ &m) {
    m.def("velocities_fwd", &velocities_fwd,
        py::arg("q1"), py::arg("t1"), py::arg("q2"), py::arg("t2"),
        py::arg("left_time"), py::arg("right_time")
    );
    m.def("velocities_bwd", &velocities_bwd,
        py::arg("q1"), py::arg("t1"), py::arg("q2"), py::arg("t2"),
        py::arg("left_time"), py::arg("right_time"),
        py::arg("d_linear"), py::arg("d_angular")
        );

    m.def("apply_velocity_fwd", &apply_velocity_fwd,
        py::arg("q0"), py::arg("t0"), py::arg("v"), py::arg("w"), py::arg("delta_t")
    );
    m.def("apply_velocity_bwd", &apply_velocity_bwd,
        py::arg("q0"), py::arg("t0"), py::arg("v"), py::arg("w"), py::arg("delta_t"),
        py::arg("grad_q"), py::arg("grad_t")
    );

    m.def("linear_motion_transform_fwd", &linear_motion_transform_fwd,
        py::arg("q0"), py::arg("t0"), py::arg("v"), py::arg("w"), py::arg("pose_time"),
        py::arg("points"), py::arg("points_time")
        );
    m.def("linear_motion_transform_bwd", &linear_motion_transform_bwd);

    m.def("linear_velocity_fwd", &linear_velocity_fwd,
        py::arg("t1"), py::arg("t2"), py::arg("dt")
    );
    m.def("linear_velocity_bwd", &linear_velocity_bwd,
        py::arg("t1"), py::arg("t2"), py::arg("dt"), py::arg("d_out")
    );

    m.def("angular_velocity_fwd", &angular_velocity_fwd,
        py::arg("q1"), py::arg("q2"), py::arg("dt")
    );
    m.def("angular_velocity_bwd", &angular_velocity_bwd,
        py::arg("q1"), py::arg("q2"), py::arg("dt"), py::arg("d_out")
    );

    m.def("apply_angular_velocity_fwd", &apply_angular_velocity_fwd,
        py::arg("q0"), py::arg("w"), py::arg("delta_t")
    );
    m.def("apply_angular_velocity_bwd", &apply_angular_velocity_bwd,
        py::arg("q0"), py::arg("w"), py::arg("delta_t"), py::arg("d_quat")
    );
}
