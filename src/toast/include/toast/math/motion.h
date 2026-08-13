#ifndef PYTOAST_MOTION_H
#define PYTOAST_MOTION_H

#include "se3.h"


template <typename dtype>
HOSTDEVICE auto linear_velocity(
    const typename dtype::vec3& t1,
    const typename dtype::vec3& t2,
    const typename dtype::T& dt
) {
    return (t2 - t1) / dt;
}

template <typename dtype>
HOSTDEVICE auto linear_velocity_backward(
    const typename dtype::vec3& t1,
    const typename dtype::vec3& t2,
    const typename dtype::T& dt,
    const typename dtype::vec3& d_velocity
) {
    auto inv_dt = typename dtype::T(1.0) / dt;
    auto inv_dt2 = inv_dt * inv_dt;

    auto diff = t2 - t1;

    // gradients
    return std::tuple{
        -d_velocity * inv_dt,
        d_velocity * inv_dt,
        -dot(d_velocity, diff) * inv_dt2
    };
}

template <typename dtype>
HOSTDEVICE auto angular_velocity_short_arc(
    const quat<dtype>& q1,
    const quat<dtype>& q2,
    const typename dtype::T& dt
) {
    return q2.quat_mul(q1.inverse()).std().to_axis_angle() / dt;
}

template <typename dtype>
HOSTDEVICE auto angular_velocity_short_arc_backward(
    const quat<dtype>& q1,
    const quat<dtype>& q2,
    const typename dtype::T& dt,
    const typename dtype::vec3& d_velocity
) {
    const auto q1_inv = q1.inverse();
    const auto rel_quat = q2.quat_mul(q1_inv);
    const auto std_quat = rel_quat.std();

    const auto dq_std = quat<dtype>::to_axis_angle_backward(std_quat, d_velocity / dt);

    const auto drel_quat = quat<dtype>::std_backward(rel_quat, dq_std);

    typename dtype::vec4 dq2, dq1_inv;
    quat<dtype>::quat_mul_backward(q2, q1_inv, drel_quat, dq2, dq1_inv);
    const auto dq1 = quat<dtype>::inverse_backward(q1, dq1_inv);

    const auto axis_angle = rel_quat.std().to_axis_angle();
    return std::tuple{dq1, dq2, -dot(d_velocity, axis_angle) / (dt * dt)};
}


template <typename dtype>
HOSTDEVICE auto estimate_velocities(
    const quat<dtype>& q1,
    const typename dtype::vec3& t1,
    const quat<dtype>& q2,
    const typename dtype::vec3& t2,
    const typename dtype::T& time1,
    const typename dtype::T& time2
) {
    const auto dt = time2 - time1;
    return std::tuple{
        linear_velocity<dtype>(t1, t2, dt),
        angular_velocity_short_arc<dtype>(q1, q2, dt)
    };
}

template <typename dtype>
HOSTDEVICE auto estimate_velocities_backward(
    const quat<dtype>& q1,
    const typename dtype::vec3& t1,
    const quat<dtype>& q2,
    const typename dtype::vec3& t2,
    const typename dtype::T& time1,
    const typename dtype::T& time2,

    const typename dtype::vec3& d_linear,
    const typename dtype::vec3& d_angular
) {
    const auto time_delta = time2 - time1;
    const auto [dt1, dt2, d_time_delta_1] = linear_velocity_backward<dtype>(t1, t2, time_delta, d_linear);
    const auto [dq1, dq2, d_time_delta_2] = angular_velocity_short_arc_backward<dtype>(q1, q2, time_delta, d_angular);
    const auto d_time_delta = d_time_delta_1 + d_time_delta_2;
    return std::tuple{dq1, dt1, dq2, dt2, -d_time_delta, d_time_delta};
}


//  apply_velocity: SE3(Quat.from_axis_angle(angular * delta_t) @ trf.q, trf.t + delta_t * linear)

template <typename dtype>
HOSTDEVICE quat<dtype> apply_angular_velocity(
    const quat<dtype>& q0,
    const typename dtype::vec3& w,
    const typename dtype::T& delta_t
) {
    return quat<dtype>::from_axis_angle(w * delta_t).quat_mul(q0);
}


template <typename dtype>
HOSTDEVICE auto apply_angular_velocity_backward(
    const quat<dtype>& q0,
    const typename dtype::vec3& w,
    const typename dtype::T& delta_t,
    const typename dtype::vec4& d_quat
) {
    const auto w_delta_t = w * delta_t;
    const auto q_w = quat<dtype>::from_axis_angle(w_delta_t);

    typename dtype::vec4 dq_w, dq0;
    quat<dtype>::quat_mul_backward(q_w, q0, d_quat, dq_w, dq0);

    const auto dw_delta_t = quat<dtype>::from_axis_angle_backward(w_delta_t, dq_w);

    return std::tuple{
        dq0,
        delta_t * dw_delta_t,
        dot(dw_delta_t, w),
    };
}


template <typename dtype>
HOSTDEVICE std::tuple<quat<dtype>, typename dtype::vec3> apply_velocity(
    const quat<dtype>& q0,
    const typename dtype::vec3& t0,
    const typename dtype::vec3& v,
    const typename dtype::vec3& w,
    const typename dtype::T& delta_t
) {
    return {apply_angular_velocity(q0, w, delta_t), t0 + delta_t * v};
}


template <typename dtype>
HOSTDEVICE auto apply_velocity_backward(
    const quat<dtype>& q0,
    const typename dtype::vec3& t0,
    const typename dtype::vec3& v,
    const typename dtype::vec3& w,
    const typename dtype::T& delta_t,
    const typename dtype::vec4& d_quat,
    const typename dtype::vec3& d_t
) {

    const auto w_delta_t = w * delta_t;
    const auto q_w = quat<dtype>::from_axis_angle(w * delta_t);

    typename dtype::vec4 dq_w, dq0;
    quat<dtype>::quat_mul_backward(q_w, q0, d_quat, dq_w, dq0);

    const auto dw_delta_t = quat<dtype>::from_axis_angle_backward(w_delta_t, dq_w);

    return std::tuple{
        dq0,
        d_t,
        delta_t * d_t,
        delta_t * dw_delta_t,
        dot(dw_delta_t, w) + dot(d_t, v),
    };
}

template <typename dtype>
HOSTDEVICE typename dtype::vec3 transform_point_dynamic(
    const quat<dtype>& q0,
    const typename dtype::vec3& t0,
    const typename dtype::vec3& v,
    const typename dtype::vec3& w,
    const typename dtype::T& pose_time,
    const typename dtype::vec3& point,
    const typename dtype::T& point_time
) {
    const auto dtime = point_time - pose_time;
    const auto [q, t] = apply_velocity(q0, t0, v, w, dtime);
    return se3<dtype>(q, t).apply(point);
}

template <typename dtype>
HOSTDEVICE void transform_point_dynamic_backward(
    const quat<dtype>& q0,
    const typename dtype::vec3& t0,
    const typename dtype::vec3& v,
    const typename dtype::vec3& w,
    const typename dtype::T& pose_time,
    const typename dtype::vec3& point,
    const typename dtype::T& point_time,
    const typename dtype::vec3& d_result,

    typename dtype::vec4& d_q0,
    typename dtype::vec3& d_t0,
    typename dtype::vec3& d_v,
    typename dtype::vec3& d_w,
    typename dtype::T& d_pose_time,
    typename dtype::vec3& d_point,
    typename dtype::T& d_point_time
) {
    const auto dtime = point_time - pose_time;
    const auto [q, t] = apply_velocity(q0, t0, v, w, dtime);

    const auto [dq_dp, dt] = se3<dtype>::apply_backward(se3<dtype>(q, t), point, d_result);

    const auto [dq0, dt0, dv, dw, d_dtime] = apply_velocity_backward(q0, t0, v, w, dtime, std::get<0>(dq_dp), dt);

    d_q0 = dq0;
    d_t0 = dt0;
    d_v = dv;
    d_w = dw;
    d_pose_time = -d_dtime;
    d_point = std::get<1>(dq_dp);
    d_point_time = d_dtime;
}

#endif //PYTOAST_MOTION_H
