#ifndef PYTOAST_SCALAR_H
#define PYTOAST_SCALAR_H

#include "../utils/vector_math.h"


template <typename scalar, typename scalar2, typename scalar3, typename scalar4>
struct FloatType {
    using T = scalar;
    using vec2 = scalar2;
    using vec3 = scalar3;
    using vec4 = scalar4;
};

struct Float32: FloatType<float, float2, float3, float4> {
    static constexpr float eps = 1e-6f;

    HOSTDEVICE static float sin(float x) {return sinf(x);}
    HOSTDEVICE static float cos(float x) {return cosf(x);}
    HOSTDEVICE static void sincos(float x, float* sinx, float* cosx) {
        sincosf(x, sinx, cosx);
    }
    HOSTDEVICE static float tan(float x) {return tanf(x);}
    HOSTDEVICE static float asin(float x) {return asinf(x);}
    HOSTDEVICE static float acos(float x) {return acosf(x);}
    HOSTDEVICE static float atan(float x) {return atanf(x);}
    HOSTDEVICE static float atan2(float y, float x) {return atan2f(y, x);}

    HOSTDEVICE static float abs(float x) {return fabsf(x);}
    HOSTDEVICE static float sqrt(float x) {return sqrtf(x);}
    HOSTDEVICE static float rsqrt(float x) {return rsqrtf(x);}
    HOSTDEVICE static float pow(float x, float y) {return ::powf(x, y);}

    HOSTDEVICE static float copysign(float x, float y) {return copysignf(x, y);}
    HOSTDEVICE static float fmod(float x, float y) {return ::fmodf(x, y);}

    HOSTDEVICE static float max(float x, float y) {return fmaxf(x, y);}
    HOSTDEVICE static float min(float x, float y) {return fminf(x, y);}

    HOSTDEVICE static float floor(float x) {return ::floorf(x);}
    HOSTDEVICE static float ceil(float x) {return ::ceilf(x);}
    HOSTDEVICE static float round(float x) {return ::roundf(x);}

    HOSTDEVICE static float exp(float x) {return ::expf(x);}
    HOSTDEVICE static float exp2(float x) {return ::exp2f(x);}
    HOSTDEVICE static float log(float x) {return ::logf(x);}
    HOSTDEVICE static float log10(float x) {return ::log10f(x);}
    HOSTDEVICE static float log2(float x) {return ::log2f(x);}
};

struct Float64: FloatType<double, double2, double3, double4_32a> {
    static constexpr double eps = 1e-12;

    HOSTDEVICE static double sin(double x) {return ::sin(x);}
    HOSTDEVICE static double cos(double x) {return ::cos(x);}
    HOSTDEVICE static void sincos(double x, double* sinx, double* cosx) {
        ::sincos(x, sinx, cosx);
    }
    HOSTDEVICE static double tan(double x) {return ::tan(x);}
    HOSTDEVICE static double asin(double x) {return ::asin(x);}
    HOSTDEVICE static double acos(double x) {return ::acos(x);}
    HOSTDEVICE static double atan(double x) {return ::atan(x);}
    HOSTDEVICE static double atan2(double y, double x) {return ::atan2(y, x);}

    HOSTDEVICE static double abs(double x) {return ::fabs(x);}
    HOSTDEVICE static double sqrt(double x) {return ::sqrt(x);}
    HOSTDEVICE static double rsqrt(double x) {return 1.0 / sqrt(x);}
    HOSTDEVICE static double pow(double x, double y) {return ::pow(x, y);}

    HOSTDEVICE static double copysign(double x, double y) {return ::copysign(x, y);}
    HOSTDEVICE static double fmod(double x, double y) {return ::fmod(x, y);}

    HOSTDEVICE static double max(double x, double y) {return ::fmax(x, y);}
    HOSTDEVICE static double min(double x, double y) {return ::fmin(x, y);}

    HOSTDEVICE static double floor(double x) {return ::floor(x);}
    HOSTDEVICE static double ceil(double x) {return ::ceil(x);}
    HOSTDEVICE static double round(double x) {return ::round(x);}

    HOSTDEVICE static double exp(double x) {return ::exp(x);}
    HOSTDEVICE static double exp2(double x) {return ::exp2(x);}
    HOSTDEVICE static double log(double x) {return ::log(x);}
    HOSTDEVICE static double log10(double x) {return ::log10(x);}
    HOSTDEVICE static double log2(double x) {return ::log2(x);}
};


template <typename T>
struct vec_dtype;

// Specializations
template <>
struct vec_dtype<float> {
    using type = Float32;
};

template <>
struct vec_dtype<double> {
    using type = Float64;
};

template <typename T>
using dtype_t = typename vec_dtype<T>::type;

#endif //PYTOAST_SCALAR_H
