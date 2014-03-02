#pragma once

#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>

__device__ __forceinline__ float mean(float3 v) {
	const float fac = 1.0 / 3.0;
	return (v.x + v.y + v.z) * fac;
}

__device__ __forceinline__ void rotate_point(const mat3& R, float &x, float &y, float &z) {
	float x0 = x, y0 = y, z0 = z;
	x = R(0) * x0 + R(1) * y0 + R(2) * z0;
	y = R(3) * x0 + R(4) * y0 + R(5) * z0;
	z = R(6) * x0 + R(7) * y0 + R(8) * z0;
}

__device__ __forceinline__ void rotate_translate_point(const mat3& R, const float3& T, float &x, float &y, float &z) {
	float x0 = x, y0 = y, z0 = z;
	x = R(0) * x0 + R(1) * y0 + R(2) * z0 + T.x;
	y = R(3) * x0 + R(4) * y0 + R(5) * z0 + T.y;
	z = R(6) * x0 + R(7) * y0 + R(8) * z0 + T.z;
}

__device__ __forceinline__ float3 color2world_fast(float u, float v, float d) {
	// focal length
	// const float fx_rgb = 525.0, fy_rgb = 525.0;
	const float inv_fx_rgb = 1.0/525.0, inv_fy_rgb = 1.0/525.0;
	// for 640x480 image
	const float cx_rgb = 320.0, cy_rgb = 240.0;

	// This part is correct now.
	// Given a Kinect depth value, its depth in OpenGL coordinates
	// system must be negative.
	float depth = -d * 0.001;

	float3 res;
	// inverse mapping of projection
	res.x = -(u - cx_rgb) * depth * inv_fx_rgb;
	res.y = (v - cy_rgb) * depth * inv_fy_rgb;
	res.z = depth;
	return res;
}

__device__ __forceinline__ float3 color2world(float u, float v, float d) {
	// focal length
	const float fx_rgb = 525.0, fy_rgb = 525.0;
	// for 640x480 image
	const float cx_rgb = 320.0, cy_rgb = 240.0;

	// This part is correct now.
	// Given a Kinect depth value, its depth in OpenGL coordinates
	// system must be negative.
	float depth = -d/1000.0;

	float3 res;
	// inverse mapping of projection
	res.x = -(u - cx_rgb) * depth / fx_rgb;
	res.y = (v - cy_rgb) * depth / fy_rgb;
	res.z = depth;
	return res;
}

__device__ __forceinline__ float3 world2color(float3 p) {
	// focal length
	const float fx_rgb = 525.0, fy_rgb = 525.0;
	// for 640x480 image
	const float cx_rgb = 320.0, cy_rgb = 240.0;

	float invZ = 1.0 / p.z;
	float3 uvd;
	uvd.x = clamp(cx_rgb - p.x * fx_rgb * invZ, 0.f, 639.f);
	uvd.y = clamp(cy_rgb + p.y * fy_rgb * invZ, 0.f, 479.f);
	uvd.z = -p.z*1000.0f;
	return uvd;
}

__device__ __forceinline__ int decodeIndex(unsigned char r, unsigned char g, unsigned char b) {
	int ri = r, gi = g, bi = b;
	return (ri << 16) | (gi << 8) | bi;
}

__device__ __forceinline__ float point_to_triangle_distance(float3 p0, float3 p1, float3 p2, float3 p3, float3& hit) {
	float dist = 0;

	float3 d = p1 - p0;
	float3 e12 = p2 - p1, e13 = p3 - p1, e21 = -e12, e23 = p3 - p2, e31 = -e13, e32 = -e23;
	float3 e12n = normalize(e12), e13n = normalize(e13), e21n = -e12n, e23n = normalize(e23), e31n = -e13n, e32n = -e23n;

	float3 n = normalize(cross(e12, e13));

	float dDOTn = dot(d, n);
	float dnorm = length(d);
	float cosAlpha = dDOTn / dnorm;

	float dn = dDOTn * cosAlpha;
	float3 p0p0c = -dn * n;
	float3 p0c = p0 + p0p0c;

	float3 v1 = e21n + e31n, v2 = e12n + e32n, v3 = e13n + e23n;
	float3 p0cp1 = p1 - p0c, p0cp2 = p2 - p0c, p0cp3 = p3 - p0c;

	float3 c1 = cross(p0cp1, p0cp2), c2 = cross(p0cp2, p0cp3), c3 = cross(p0cp3, p0cp1);

	float d1 = dot(c1, n);
	float d2 = dot(c2, n);
	float d3 = dot(c3, n);

	float3 x0 = p0c, x1, x2;
	bool inside = true;
	if ( d1 < d2 && d1 < d3 && d1<0 )			//-- outside, p1, p2 side
	{
		inside = false;		
		x1 = p1; x2 = p2;
	}
	else if ( d2 < d1 && d2 < d3 && d2<0 )	//-- outside, p2, p3 side
	{
		inside = false;
		x1 = p2; x2 = p3;
	}
	else if ( d3 < d1 && d3 < d2 && d3<0 )	//-- outside, p3, p1 side
	{
		inside = false;
		x1 = p3; x2 = p1;
	}

	if (inside)
	{
		hit = p0c;
		dist = dn;
	}
	else
	{
		float3 x1x0 = x0 - x1, x2x0 = x0 - x1, x2x1 = x1 - x2;
		float L_x2x0 = length(x2x0);
		float t = dot(x1x0, x2x0) / (L_x2x0 * L_x2x0);

		hit = x1 + t * x2x1;

		float3 line = p0 - hit;
		dist = length(line);
	}

	return dist;

}

__device__ __forceinline__ float3 compute_barycentric_coordinates(float3 p, float3 q1, float3 q2, float3 q3) {
	float3 e23 = q3 - q2, e21 = q1 - q2, e31 = q1 - q3;
	float3 d2 = p - q2, d3 = p - q3;
	float3 oriN = cross(e23, e21);
	float3 n = normalize(oriN);

	float invBTN = 1.0 / dot(oriN, n);
	float3 bcoord;
	bcoord.x = dot(cross(e23, d2), n) * invBTN;
	bcoord.y = dot(cross(e31, d3), n) * invBTN;
	bcoord.z = 1 - bcoord.x - bcoord.y;

	return bcoord;
}
