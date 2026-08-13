#ifndef PYTOAST_BWD_MATH_UTILS_H
#define PYTOAST_BWD_MATH_UTILS_H

#include "vector_math.h"
#include "atomics.h"



template <typename dtype, typename T>
HOSTDEVICE T normalize_bwd(const T& v, const T& d_out) {
    using S = typename dtype::T;
    const S v_len = dot(v, v);
    if (v_len > S(0.0)) {
        const S inv_len = dtype::rsqrt(v_len);
        return inv_len * (d_out - inv_len * inv_len * dot(d_out, v) * v);
    }
    return d_out;
}


template <typename T, typename scalar_t>
HOSTDEVICE std::tuple<T, T, scalar_t> lerp_backward(
    const T& a, const T& b, const scalar_t& w, const T& grad_out
) {
    return {(1.0f - w) * grad_out, w * grad_out, dot(grad_out, (b - a))};
}


template <typename dtype>
HOSTDEVICE void atomicAdd(
    typename dtype::vec3* addr,
    const typename dtype::vec3& val
) {
    toast_atomic_add(&(addr->x), val.x);
    toast_atomic_add(&(addr->y), val.y);
    toast_atomic_add(&(addr->z), val.z);
}

template <typename dtype>
HOSTDEVICE void atomicAdd(
    typename dtype::vec4* addr,
    const typename dtype::vec4& val
) {
    toast_atomic_add(&(addr->x), val.x);
    toast_atomic_add(&(addr->y), val.y);
    toast_atomic_add(&(addr->z), val.z);
    toast_atomic_add(&(addr->w), val.w);
}



#endif //PYTOAST_BWD_MATH_UTILS_H
