#ifndef PYTOAST_HOST_VECTOR_TYPES_H
#define PYTOAST_HOST_VECTOR_TYPES_H


#include <math.h>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif


struct __align__(8)  int2   { int x, y; };
struct               int3   { int x, y, z; };
struct __align__(16) int4   { int x, y, z, w; };

struct __align__(8)  uint2  { unsigned int x, y; };
struct               uint3  { unsigned int x, y, z; };
struct __align__(16) uint4  { unsigned int x, y, z, w; };

struct __align__(8)  float2 { float x, y; };
struct               float3 { float x, y, z; };
struct __align__(16) float4 { float x, y, z, w; };

struct __align__(16) double2 { double x, y; };
struct               double3 { double x, y, z; };
struct __align__(16) double4 { double x, y, z, w; };
struct __align__(16) double4_32a { double x, y, z, w; };


static __forceinline__ int2   make_int2(int x, int y)                       { return int2{x, y}; }
static __forceinline__ int3   make_int3(int x, int y, int z)                { return int3{x, y, z}; }
static __forceinline__ int4   make_int4(int x, int y, int z, int w)         { return int4{x, y, z, w}; }

static __forceinline__ uint2  make_uint2(unsigned int x, unsigned int y)    { return uint2{x, y}; }
static __forceinline__ uint3  make_uint3(unsigned int x, unsigned int y, unsigned int z)
                                                                            { return uint3{x, y, z}; }
static __forceinline__ uint4  make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
                                                                            { return uint4{x, y, z, w}; }

static __forceinline__ float2 make_float2(float x, float y)                 { return float2{x, y}; }
static __forceinline__ float3 make_float3(float x, float y, float z)        { return float3{x, y, z}; }
static __forceinline__ float4 make_float4(float x, float y, float z, float w)
                                                                            { return float4{x, y, z, w}; }

static __forceinline__ double2 make_double2(double x, double y)             { return double2{x, y}; }
static __forceinline__ double3 make_double3(double x, double y, double z)   { return double3{x, y, z}; }
static __forceinline__ double4 make_double4(double x, double y, double z, double w)
                                                                            { return double4{x, y, z, w}; }
static __forceinline__ double4_32a make_double4_32a(double x, double y, double z, double w)
                                                                            { return double4_32a{x, y, z, w}; }

#if !defined(__GLIBC__) && !defined(TOAST_HAVE_SINCOS)
static __forceinline__ void sincosf(float x, float* s, float* c) { *s = sinf(x); *c = cosf(x); }
static __forceinline__ void sincos(double x, double* s, double* c) { *s = sin(x); *c = cos(x); }
#endif

static_assert(sizeof(int2) == 8    && alignof(int2) == 8,     "int2 layout differs from CUDA");
static_assert(sizeof(int3) == 12   && alignof(int3) == 4,     "int3 layout differs from CUDA");
static_assert(sizeof(int4) == 16   && alignof(int4) == 16,    "int4 layout differs from CUDA");
static_assert(sizeof(uint2) == 8   && alignof(uint2) == 8,    "uint2 layout differs from CUDA");
static_assert(sizeof(uint3) == 12  && alignof(uint3) == 4,    "uint3 layout differs from CUDA");
static_assert(sizeof(uint4) == 16  && alignof(uint4) == 16,   "uint4 layout differs from CUDA");
static_assert(sizeof(float2) == 8  && alignof(float2) == 8,   "float2 layout differs from CUDA");
static_assert(sizeof(float3) == 12 && alignof(float3) == 4,   "float3 layout differs from CUDA");
static_assert(sizeof(float4) == 16 && alignof(float4) == 16,  "float4 layout differs from CUDA");
static_assert(sizeof(double2) == 16 && alignof(double2) == 16, "double2 layout differs from CUDA");
static_assert(sizeof(double3) == 24 && alignof(double3) == 8,  "double3 layout differs from CUDA");
static_assert(sizeof(double4) == 32 && alignof(double4) == 16, "double4 layout differs from CUDA");
static_assert(sizeof(double4_32a) == 32, "double4_32a must be 4 doubles wide");

#endif // PYTOAST_HOST_VECTOR_TYPES_H
