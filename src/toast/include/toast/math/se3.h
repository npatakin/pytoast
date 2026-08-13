#ifndef PYTOAST_SE3_H
#define PYTOAST_SE3_H
#include "quat.h"


template <typename dtype = Float32>
struct se3 {
    using T = typename dtype::T;
    using vec2 = typename dtype::vec2;
    using vec3 = typename dtype::vec3;
    using vec4 = typename dtype::vec4;
    using quat_t = quat<dtype>;
    
    quat_t q;
    vec3 t;

    HOSTDEVICE explicit se3(): q(), t({T(0.0), T(0.0), T(0.0)}) {}
    HOSTDEVICE explicit se3(const quat_t& q, const vec3& t): q(q), t(t) {}

    [[nodiscard]] HOSTDEVICE vec3 apply(const vec3& p) const {
        return q.apply(p) + t;
    }

    HOSTDEVICE static std::tuple<std::tuple<vec4, vec3>, vec3> apply_backward(
        const se3& trf, const vec3& p, const vec3& grad_out
    ) {
        return {quat_t::apply_backward(trf.q, p, grad_out), grad_out};
    }

    [[nodiscard]] HOSTDEVICE vec3 apply_inv(const vec3& p) const {
        return q.apply_inv(p - t);
    }

    HOSTDEVICE static std::tuple<std::tuple<vec4, vec3>, vec3> apply_inv_backward(
        const se3& trf, const vec3& p, const vec3& grad_out
    ) {
        const vec3 v = p - trf.t;
        const auto [dq, dv] = quat_t::apply_inv_backward(trf.q, v, grad_out);
        return {{dq, dv}, -dv};
    }

    [[nodiscard]] HOSTDEVICE se3 inverse() const {
        const auto q_inv = q.inverse();
        return se3{q_inv, -T(1.0) * q_inv.apply(t)};
    }

    HOSTDEVICE static std::tuple<vec4, vec3> inverse_backward(
        const se3& trf, const vec4& grad_quat_out, const vec3& grad_t_out
    ) {
        const auto q_inv = trf.q.inverse();

        auto [d_q_inv, d_t] = quat_t::apply_backward(q_inv, trf.t, -grad_t_out);
        d_q_inv += grad_quat_out;

        return {quat_t::inverse_backward(trf.q, d_q_inv), d_t};
    }

    [[nodiscard]] HOSTDEVICE se3 compose(const se3& other) const {
        return se3{q.quat_mul(other.q), this->apply(other.t)};
    }

    HOSTDEVICE static std::tuple<vec4, vec3, vec4, vec3> compose_backward(
        const se3& a, const se3& b, const vec4& grad_quat_out, const vec3& grad_t_out
    ) {
        vec4 dq1, dq2;
        quat_t::quat_mul_backward(a.q, b.q, grad_quat_out, dq1, dq2);

        const auto [da, dt2] = apply_backward(a, b.t, grad_t_out);
        return {dq1 + std::get<0>(da), dt2, dq2, std::get<1>(da)};
    }

    [[nodiscard]] HOSTDEVICE se3 slerp(const se3& other, const T w) const {
        return se3{q.slerp(other.q, w), lerp(t, other.t, w)};
    }

    HOSTDEVICE static std::tuple<vec4, vec3, vec4, vec3, T> slerp_backward(
        const se3& a, const se3& b, const T w, const se3& grad_out
    ) {
        const auto [dq1, dq2, dw_1] = quat_t::slerp_backward(
            a.q.q, b.q.q, w, grad_out.q.q
        );
        const auto [dt1, dt2, dw_2] = lerp_backward(a.t, b.t, w, grad_out.t);
        return {dq1, dt1, dq2, dt2, dw_1 + dw_2};
    }

    HOSTDEVICE void to_matrix_3x4(vec4* out) const {
        q.to_matrix(out);
        out[0].w = t.x;
        out[1].w = t.y;
        out[2].w = t.z;
    }

    HOSTDEVICE static std::tuple<vec4, vec3> to_matrix_3x4_backward(const se3& trf, const vec4* grad_out) {
        return {
            quat_t::to_matrix_backward(trf.q, grad_out),
            {grad_out[0].w, grad_out[1].w, grad_out[2].w}
        };
    }

    HOSTDEVICE void to_matrix_4x4(vec4* out) const {
        to_matrix_3x4(out);
        out[3] = {T(0.0), T(0.0), T(0.0), T(1.0)};
    }

    HOSTDEVICE static std::tuple<vec4, vec3> to_matrix_4x4_backward(const se3& trf, const vec4* grad_out) {
        return to_matrix_3x4_backward(trf, grad_out);
    }


    HOSTDEVICE static se3 from_matrix(const vec4* mtx) {
        return se3{quat_t::from_matrix(mtx), {mtx[0].w, mtx[1].w, mtx[2].w}};
    }

    HOSTDEVICE static void from_matrix_backward(const vec4* mtx, const vec4& grad_quat_out, const vec3& grad_t_out, vec4* out) {
        quat_t::from_matrix_backward(mtx, grad_quat_out, out);
        out[0].w = grad_t_out.x;
        out[1].w = grad_t_out.y;
        out[2].w = grad_t_out.z;
    }
};


using se3f = se3<Float32>;
using se3d = se3<Float64>;


#endif //PYTOAST_SE3_H
