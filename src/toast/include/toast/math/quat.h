#ifndef PYTOAST_QUAT_H
#define PYTOAST_QUAT_H

#include <tuple>

#include "../utils/vector_math.h"
#include "../utils/bwd_math_utils.h"

#include "scalar.h"


template <typename dtype = Float32>
struct quat {
    using T = typename dtype::T;
    using vec2 = typename dtype::vec2;
    using vec3 = typename dtype::vec3;
    using vec4 = typename dtype::vec4;

    vec4 q;

    explicit HOSTDEVICE quat(const T w, const T x, const T y, const T z): q({w, x, y, z}) {}
    explicit HOSTDEVICE quat(const vec4& q):q(q) {}
    explicit HOSTDEVICE quat(const T w, const vec3& xyz): q({w, xyz.x, xyz.y, xyz.z}) {}
    explicit HOSTDEVICE quat(): quat(T(1.0), T(0.0), T(0.0), T(0.0)) {}


    [[nodiscard]] HOSTDEVICE static quat from_xyzw(const T x, const T y, const T z, const T w) {
        return quat(w, x, y, z);
    }

    [[nodiscard]] HOSTDEVICE static vec4 from_xyzw_backward(const vec4& grad_out) {
        return {grad_out.y, grad_out.z, grad_out.w, grad_out.x};
    }

    [[nodiscard]] HOSTDEVICE static quat random_from_uniform(const vec3& r) {
        const auto& [r1, r2, r3] = r;

        const auto a = dtype::sqrt(T(1.0) - r1);
        const auto b = dtype::sqrt(r1);

        T sin_r2, cos_r2;
        dtype::sincos(r2 * T(2.0 * M_PI), &sin_r2, &cos_r2);
        T sin_r3, cos_r3;
        dtype::sincos(r3 * T(2.0 * M_PI), &sin_r3, &cos_r3);

        return quat(a * sin_r2, a * cos_r2, b * sin_r3, b * cos_r3);
    }

    [[nodiscard]] HOSTDEVICE T real() const {return q.x;}
    [[nodiscard]] HOSTDEVICE vec3 img() const {return {q.y, q.z, q.w};}

    [[nodiscard]] HOSTDEVICE quat std() const {
        return quat{normalize(dtype::copysign(T(1.0), real()) * q)};
    }

    static HOSTDEVICE vec4 std_backward(const quat& input, const vec4& grad_out) {
        T s = dtype::copysign(T(1.0), input.real());
        return s * normalize_bwd<dtype>(input.q * s, grad_out);
    }

    [[nodiscard]] HOSTDEVICE quat conjugate() const {
        return quat(real(), -img());
    }

    static HOSTDEVICE vec4 conjugate_backward(const vec4& grad_out) {
        return {grad_out.x, -grad_out.y, -grad_out.z, -grad_out.w};
    }

    [[nodiscard]] HOSTDEVICE quat inverse() const {
        const auto inv_norm2 = T(1.0) / dot(q, q);
        return quat(real() * inv_norm2, -img() * inv_norm2);
    }

    [[nodiscard]] HOSTDEVICE static vec4 inverse_backward(const quat& q, const vec4& grad_out) {
        const auto norm2 = dot(q.q, q.q);
        const auto inv_s = T(1.0) / norm2;
        const auto dot_term = -T(2.0) * dot(grad_out, q.conjugate().q) / (norm2 * norm2);
        return conjugate_backward(grad_out) * inv_s + dot_term * q.q;
    }


    HOSTDEVICE quat quat_mul(const quat& other) const {
        const auto& [w1, x1, y1, z1] = q;
        const auto& [w2, x2, y2, z2] = other.q;

        return quat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
             w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        );
    }

    static HOSTDEVICE void
    quat_mul_backward(const quat& a, const quat& b, const vec4& grad_w, vec4& d_q1, vec4& d_q2) {
        const auto& [aw, ax, ay, az] = a.q;
        const auto& [bw, bx, by, bz] = b.q;
        const auto& [g0, g1, g2, g3] = grad_w;

        d_q1 = {
            g0*bw + g1*bx + g2*by + g3*bz,
               -g0*bx + g1*bw - g2*bz + g3*by,
               -g0*by + g1*bz + g2*bw - g3*bx,
               -g0*bz - g1*by + g2*bx + g3*bw
        };
        d_q2 = {
            g0*aw + g1*ax + g2*ay + g3*az,
            -g0*ax + g1*aw + g2*az - g3*ay,
            -g0*ay - g1*az + g2*aw + g3*ax,
            -g0*az + g1*ay - g2*ax + g3*aw
        };
    }


    [[nodiscard]] HOSTDEVICE vec3 apply(const vec3& v) const {
        const auto uv = cross(img(), v);
        const auto uuv = cross(img(), uv);
        return v + T(2.0) * (real() * uv + uuv);
    }

    static HOSTDEVICE std::tuple<vec4, vec3> apply_backward(
        const quat& q, const vec3& v, const vec3& grad_out
    ) {
        const vec3 img = q.img();
        const vec3 uv  = cross(img, v);

        const vec3 g_cross_img = cross(grad_out, img);
        const vec3 dv = grad_out + T(2.0) * (q.real() * g_cross_img + cross(g_cross_img, img));
        const vec3 d_img = T(2.0) * (
            q.real() * cross(v, grad_out)
            + cross(uv, grad_out)
            + cross(v, g_cross_img)
        );
        const auto dw = T(2.0) * dot(grad_out, uv);
        return {{dw, d_img.x, d_img.y, d_img.z}, dv};
    }


    [[nodiscard]] HOSTDEVICE vec3 apply_inv(const vec3& v) const {
        const auto inv_norm2 = T(1.0) / dot(q, q);
        const auto inv_img = (-inv_norm2) * img();
        const auto uv = cross(inv_img, v);
        const auto uuv = cross(inv_img, uv);
        return v + T(2.0) * ((real() * inv_norm2) * uv + uuv);
    }

    [[nodiscard]] HOSTDEVICE static std::tuple<vec4, vec3> apply_inv_backward(
        const quat& q, const vec3& v, const vec3& grad_out
    ) {
        const quat q_inv = q.inverse();
        const auto [d_qinv, d_v] = apply_backward(q_inv, v, grad_out);
        const vec4 d_q = inverse_backward(q, d_qinv);
        return {d_q, d_v};
    }


    [[nodiscard]] HOSTDEVICE quat slerp(const quat& other, const T& t) const {
        vec4 z = other.q;
        T cosTheta = dot(q, z);
        if (cosTheta < T(0)) {
            z = -z;
            cosTheta = -cosTheta;
        }

        vec4 result;
        if (cosTheta >= (T(1.0) - dtype::eps)) {
            result = lerp(q, z, t);
        } else {
            T angle = dtype::acos(cosTheta);
            T denom = dtype::sin(angle);
            T w1 = dtype::sin((T(1.0) - t) * angle) / denom;
            T w2 = dtype::sin(t * angle) / denom;
            result = w1 * q + w2 * z;
        }

        return quat(result).std();
    }

    static HOSTDEVICE std::tuple<vec4, vec4, T> slerp_backward(
        const vec4& q1,
        const vec4& q2,
        const T t,
        const vec4& grad_out)
    {
        vec4 z = q2;
        T cosTheta = dot(q1, z);

        bool flipped = false;
        if (cosTheta < T(0)) {
            flipped = true;
            z = -z;
            cosTheta = -cosTheta;
        }

        vec4 dq1, dq2; T dt;

        if (cosTheta >= (T(1.0) - dtype::eps)) {
            vec4 grad_result = std_backward(quat{lerp(q1, z, t)}, grad_out);
            dq1 = (T(1.0) - t) * grad_result;
            dq2 = t * grad_result;
            dt  = dot(grad_result, (z - q1));
        } else {
            const T angle = dtype::acos(cosTheta);
            const T sin_theta = dtype::sin(angle);
            const T inv_sin = T(1.0) / sin_theta;

            T w1_sin, w1_cos;
            dtype::sincos((T(1.0) - t) * angle, &w1_sin, &w1_cos);

            const T w1 = w1_sin * inv_sin;
            const T w2 = dtype::sin(t * angle) * inv_sin;

            vec4 grad_result = std_backward(quat{w1 * q1 + w2 * z}, grad_out);
            
            const T dw1 = dot(grad_result, q1);
            const T dw2 = dot(grad_result, z);
            
            dq1 = w1 * grad_result;
            dq2 = w2 * grad_result;
            
            const T cos_ta  = dtype::cos(t * angle);
            const T dw1_dtheta = ( (T(1.0) - t) * w1_cos * sin_theta - w1_sin * dtype::cos(angle) ) / (sin_theta * sin_theta);
            const T dw2_dtheta = ( t * cos_ta * sin_theta - dtype::sin(t * angle) * dtype::cos(angle) ) / (sin_theta * sin_theta);
            const T dL_dtheta = dw1 * dw1_dtheta + dw2 * dw2_dtheta;

            const T dtheta_dcos = -dtype::rsqrt(T(1.0) - cosTheta * cosTheta);
            const T dL_dcos = dL_dtheta * dtheta_dcos;
            
            dq1 += dL_dcos * z;
            dq2 += dL_dcos * q1;
            
            const T dw1_dt = -angle * w1_cos * inv_sin;
            const T dw2_dt =  angle * cos_ta  * inv_sin;

            dt = dw1 * dw1_dt + dw2 * dw2_dt;
        }
        if (flipped) dq2 = -dq2;

        return {dq1, dq2, dt};
    }





    [[nodiscard]] HOSTDEVICE vec3 to_axis_angle() const {
        const vec3 v = img(); const T w = real();
        const T s = dtype::sqrt(dot(v, v));

        const T angle = T(2.0) * dtype::atan2(s, w);
        const T scale = (s > dtype::eps) ? (angle / s) : T(2.0);
        return v * scale;
    }

    static HOSTDEVICE vec4 to_axis_angle_backward(const quat& q, const vec3& grad_out) {
        const vec3 v = q.img(); const T w = q.real();
        const T s2 = dot(v, v);
        const T s = dtype::sqrt(s2);

        if (s < dtype::eps) {
            return {T(0), T(2.0)*grad_out.x, T(2.0) * grad_out.y, T(2.0) * grad_out.z};
        }

        const T angle = T(2.0) * dtype::atan2(s, w);
        const T denom = s2 + w * w;

        const T dangle_ds = T(2.0) * w / denom;
        const T dangle_dw = -T(2.0) * s / denom;

        const T dalpha_ds = (dangle_ds / s) - (angle / s2);
        const T dot_g_v = dot(grad_out, v);
        const vec3 grad_v = (angle / s) * grad_out + dalpha_ds * (dot_g_v / s) * v;

        return {dangle_dw * dot_g_v / s, grad_v.x, grad_v.y, grad_v.z};
    }


    HOSTDEVICE static quat from_axis_angle(const vec3& axis_angle) {
        T theta2 = dot(axis_angle, axis_angle);
        T theta = dtype::sqrt(theta2);

        T sin_half, cos_half;
        dtype::sincos(T(0.5) * theta, &sin_half, &cos_half);

        T scale = (theta < dtype::eps) ? T(0.5) - theta2 * (T(1.0) / T(48.0)) : sin_half / theta;
        return quat(cos_half, scale * axis_angle);
    }

    HOSTDEVICE static vec3 from_axis_angle_backward(const vec3& axis_angle, const vec4& grad_out) {
        T theta2 = dot(axis_angle, axis_angle);
        T theta = dtype::sqrt(theta2);

        const T gw = grad_out.x;
        const vec3 gv{grad_out.y, grad_out.z, grad_out.w};

        const T dot_gv_v = dot(gv, axis_angle);

        if (theta < dtype::eps) {
            const vec3 grad_vec =
                (T(0.5) - theta2 * (T(1.0) / T(48.0))) * gv - (T(1.0) / T(24.0)) * dot_gv_v * axis_angle;

            return (-T(0.25) * gw) * axis_angle + grad_vec;
        }

        T sin_half, cos_half;
        dtype::sincos(T(0.5) * theta, &sin_half, &cos_half);

        T inv_theta = T(1.0) / theta;
        T alpha = sin_half * inv_theta;

        vec3 grad_w = (-T(0.5) * sin_half * inv_theta) * axis_angle * gw;
        T d_alpha_dtheta = (T(0.5) * cos_half * theta - sin_half) / (theta * theta);
        vec3 grad_vec = alpha * gv + d_alpha_dtheta * (dot_gv_v * inv_theta) * axis_angle;
        return grad_w + grad_vec;
    }





    template <typename vec_type>
    HOSTDEVICE void to_matrix(vec_type* out) const {
        const auto& [w, x, y, z] = q;
        const auto two_s = T(2.0) / dot(q, q);
        auto& r0 = out[0];
        r0.x = T(1.0) - two_s * (y * y + z * z);
        r0.y = two_s*(x*y - z*w);
        r0.z = two_s*(x*z + y*w);

        auto& r1 = out[1];
        r1.x = two_s*(x*y + z*w);
        r1.y = T(1.0) - two_s * (x*x + z*z);
        r1.z = two_s * (y*z - x*w);

        auto& r2 = out[2];
        r2.x = two_s*(x*z - y*w);
        r2.y = two_s*(y*z + x*w);
        r2.z = T(1.0) - two_s*(x*x + y*y);
    }


    template <typename vec_type>
    HOSTDEVICE static vec4 to_matrix_backward(const quat& q, const vec_type* grad_out) {
        const auto& [w, x, y, z] = q.q;
        const auto s = dot(q.q, q.q);
        const auto inv_s = T(2.0) / s;

        // const auto& [g00, g01, g02] = grad_out[0];
        const auto& g00 = grad_out[0].x, g01 = grad_out[0].y, g02 = grad_out[0].z;
        vec4 dq{
            -z * g01 + y * g02,
            y * g01 + z * g02,
            -T(2.0) * y * g00 + x * g01 + w * g02,
            -T(2.0) * z * g00 - w * g01 + x * g02
        };
        auto accum = -(y*y + z*z) * g00 + (x*y - z*w) * g01 + (x*z + y*w) * g02;

        // const auto& [g10, g11, g12] = grad_out[1];
        const auto& g10 = grad_out[1].x, g11 = grad_out[1].y, g12 = grad_out[1].z;
        accum += (x*y + z*w) * g10 -(x*x + z*z) * g11 + (y*z - x*w) * g12;
        dq += {
            z * g10 - x * g12,
            y * g10 - T(2.0) * x * g11 - w * g12,
            x * g10 + z * g12,
            w * g10 - T(2.0) * z * g11 + y * g12
        };

        // const auto& [g20, g21, g22] = grad_out[2];
        const auto& g20 = grad_out[2].x, g21 = grad_out[2].y, g22 = grad_out[2].z;
        accum += (x*z - y*w) * g20 + (y*z + x*w) * g21 -(x*x + y*y) * g22;
        dq += {
            - y * g20 + x * g21,
            z * g20 + w * g21 - T(2.0) * x * g22,
             - w * g20 + z * g21 - T(2.0) * y * g22,
            x * g20 + y * g21
        };

        return dq * inv_s - (T(4.0) * accum / (s * s)) * q.q;
    }



    // if (trace > 0) {
    // } else if ((m00 > m11) && (m00 > m22)) {
    // } else if (m11 > m22) {
    // else {

    template <typename vec_type>
    HOSTDEVICE static quat from_matrix(const vec_type* m) {
        const auto m00 = m[0].x, m01 = m[0].y, m02 = m[0].z;
        const auto m10 = m[1].x, m11 = m[1].y, m12 = m[1].z;
        const auto m20 = m[2].x, m21 = m[2].y, m22 = m[2].z;
        const auto trace = m00 + m11 + m22;

        const auto s0 = trace + T(1.0);
        const auto s1 = T(1.0) + m00 - m11 - m22;
        const auto s2 = T(1.0) - m00 + m11 - m22;
        const auto s3 = T(1.0) - m00 - m11 + m22;
        const auto s_max = dtype::max(dtype::max(s0, s1), dtype::max(s2, s3));

        const auto s = T(2.0) * dtype::sqrt(s_max);
        const auto inv_s = T(1.0) / s;

        if (s0 == s_max) {
            return quat(T(0.25) * s, vec3{m21 - m12, m02 - m20, m10 - m01} * inv_s);
        }
        if (s1 == s_max) {
            return quat((m21 - m12) * inv_s, T(0.25) * s, (m01 + m10) * inv_s, (m02 + m20) * inv_s);
        }
        if (s2 == s_max) {
            return quat((m02 - m20) * inv_s, (m01 + m10) * inv_s, T(0.25) * s, (m12 + m21) * inv_s);
        }
        return quat((m10 - m01) * inv_s, (m02 + m20) * inv_s, (m12 + m21) * inv_s, T(0.25) * s);
    }


    template <typename vec_type>
    HOSTDEVICE static void from_matrix_backward(
        const vec_type* mtx,
        const vec4& grad_out,
        vec_type* d_mtx
    ) {
        const auto m00 = mtx[0].x, m01 = mtx[0].y, m02 = mtx[0].z;
        const auto m10 = mtx[1].x, m11 = mtx[1].y, m12 = mtx[1].z;
        const auto m20 = mtx[2].x, m21 = mtx[2].y, m22 = mtx[2].z;

        const auto s0 = m00 + m11 + m22 + T(1.0);
        const auto s1 = T(1.0) + m00 - m11 - m22;
        const auto s2 = T(1.0) - m00 + m11 - m22;
        const auto s3 = T(1.0) - m00 - m11 + m22;
        const auto s_max = dtype::max(dtype::max(s0, s1), dtype::max(s2, s3));

        const auto s = T(2.0) * dtype::sqrt(s_max);
        const auto inv_s = T(1.0) / s;
        const auto inv_s3 = inv_s * inv_s * inv_s;

        const auto gw = grad_out.x, gx = grad_out.y, gy = grad_out.z, gz = grad_out.w;

        T dm00, dm01, dm02, dm10, dm11, dm12, dm20, dm21, dm22;
        T D, c;

        if (s0 == s_max) {
            D = gx*(m21-m12) + gy*(m02-m20) + gz*(m10-m01);
            c = T(0.5)*gw*inv_s - T(2.0)*D*inv_s3;
            dm00 = +c;           dm01 = -gz*inv_s;    dm02 = +gy*inv_s;
            dm10 = +gz*inv_s;    dm11 = +c;           dm12 = -gx*inv_s;
            dm20 = -gy*inv_s;    dm21 = +gx*inv_s;    dm22 = +c;
        } else if (s1 == s_max) {
            D = gw*(m21-m12) + gy*(m01+m10) + gz*(m02+m20);
            c = T(0.5)*gx*inv_s - T(2.0)*D*inv_s3;
            dm00 = +c;           dm01 = +gy*inv_s;    dm02 = +gz*inv_s;
            dm10 = +gy*inv_s;    dm11 = -c;           dm12 = -gw*inv_s;
            dm20 = +gz*inv_s;    dm21 = +gw*inv_s;    dm22 = -c;
        } else if (s2 == s_max) {
            D = gw*(m02-m20) + gx*(m01+m10) + gz*(m12+m21);
            c = T(0.5)*gy*inv_s - T(2.0)*D*inv_s3;
            dm00 = -c;           dm01 = +gx*inv_s;    dm02 = +gw*inv_s;
            dm10 = +gx*inv_s;    dm11 = +c;           dm12 = +gz*inv_s;
            dm20 = -gw*inv_s;    dm21 = +gz*inv_s;    dm22 = -c;
        } else {
            D = gw*(m10-m01) + gx*(m02+m20) + gy*(m12+m21);
            c = T(0.5)*gz*inv_s - T(2.0)*D*inv_s3;
            dm00 = -c;           dm01 = -gw*inv_s;    dm02 = +gx*inv_s;
            dm10 = +gw*inv_s;    dm11 = -c;           dm12 = +gy*inv_s;
            dm20 = +gx*inv_s;    dm21 = +gy*inv_s;    dm22 = +c;
        }

        d_mtx[0].x = dm00;  d_mtx[0].y = dm01;  d_mtx[0].z = dm02;
        d_mtx[1].x = dm10;  d_mtx[1].y = dm11;  d_mtx[1].z = dm12;
        d_mtx[2].x = dm20;  d_mtx[2].y = dm21;  d_mtx[2].z = dm22;
    }


    HOSTDEVICE static quat from_euler_angles(const vec3& roll_pitch_yaw) {
        const auto& [roll, pitch, yaw] = roll_pitch_yaw * T(0.5);

        T sr, cr;
        dtype::sincos(roll, &sr, &cr);
        
        T sp, cp;
        dtype::sincos(pitch, &sp, &cp);
        
        T sy, cy;
        dtype::sincos(yaw, &sy, &cy);
        
        return quat(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ).std();
    }
    
    HOSTDEVICE static vec3 from_euler_angles_bwd(const vec3& roll_pitch_yaw, const vec4& grad_out) {
        const auto& [roll, pitch, yaw] = roll_pitch_yaw * T(0.5);

        T sr, cr;
        dtype::sincos(roll, &sr, &cr);
        
        T sp, cp;
        dtype::sincos(pitch, &sp, &cp);
        
        T sy, cy;
        dtype::sincos(yaw, &sy, &cy);
        
        quat quat_raw(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
        const auto d_quat_raw = quat::std_backward(quat_raw, grad_out);
      
        T dcr = dot(grad_out, vec4{cp * cy, -sp * sy, sp * cy, cp * sy});
        T dsr = dot(grad_out, vec4{sp * sy, cp * cy, cp * sy, -sp * cy});
        T dcp = dot(grad_out, vec4{cr * cy, sr * cy, sr * sy, cr * sy});
        T dsp = dot(grad_out, vec4{sr * sy, -cr * sy, cr * cy, -sr * cy});
        T dcy = dot(grad_out, vec4{cr * cp, sr * cp, cr * sp, -sr * sp});
        T dsy = dot(grad_out, vec4{sr * sp, -cr * sp, sr * cp, cr * cp});

        T droll = dsr * cr - dcr * sr;
        T dpitch = dsp * cp - dcp * sp;
        T dyaw = dsy * cy - dcy * sy;
        return T(0.5) * vec3{droll, dpitch, dyaw};
    }
    
};

using quatf = quat<Float32>;
using quatd = quat<Float64>;


#endif //PYTOAST_QUAT_H
