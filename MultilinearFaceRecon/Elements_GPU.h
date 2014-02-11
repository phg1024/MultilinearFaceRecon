#pragma once

class mat3 {
public:
	__device__ __host__ mat3(){}
	__device__ __host__ mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22) {
		elem[0] = m00; elem[1] = m01; elem[2] = m02;
		elem[3] = m10; elem[4] = m11; elem[5] = m12;
		elem[6] = m20; elem[7] = m21; elem[8] = m22;
	}
	__device__ __host__ mat3(const mat3& m){
		elem[0] = m.elem[0]; elem[1] = m.elem[1]; elem[2] = m.elem[2];
		elem[3] = m.elem[3]; elem[4] = m.elem[4]; elem[5] = m.elem[5];
		elem[6] = m.elem[6]; elem[7] = m.elem[7]; elem[8] = m.elem[8];
	}
	__device__ __host__ mat3(float *m) {
		elem[0] = m[0]; elem[1] = m[1]; elem[2] = m[2];
		elem[3] = m[3]; elem[4] = m[4]; elem[5] = m[5];
		elem[6] = m[6]; elem[7] = m[7]; elem[8] = m[8];
	}
	__device__ __host__ mat3& operator=(const mat3& m) {
		elem[0] = m.elem[0]; elem[1] = m.elem[1]; elem[2] = m.elem[2];
		elem[3] = m.elem[3]; elem[4] = m.elem[4]; elem[5] = m.elem[5];
		elem[6] = m.elem[6]; elem[7] = m.elem[7]; elem[8] = m.elem[8];
		return (*this);
	}

	__device__ __host__ static mat3 zero() {
		return mat3(0, 0, 0, 0, 0, 0, 0, 0, 0);
	}
	__device__ __host__ static mat3 identity() {
		return mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);
	}

	__device__ __host__ static mat3 rotation_x(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			1.0,	0.0,		0.0,
			0.0,	cosTheta,	-sinTheta,
			0.0,	sinTheta,	cosTheta
			);
	}

	__device__ __host__ static mat3 rotation_y(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			cosTheta,	0.0,	sinTheta,
			0.0,		1.0,	0.0,
			-sinTheta,	0.0,	cosTheta
			);
	}

	__device__ __host__ static mat3 rotation_z(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			cosTheta,	-sinTheta,	0.0,
			sinTheta,	cosTheta,	0.0,
			0.0,		0.0,		1.0
			);
	}

	__device__ __host__ static mat3 rotation_dx(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			1.0,	0.0,		0.0,
			0.0,	-sinTheta,	-cosTheta,
			0.0,	cosTheta,	-sinTheta
			);
	}

	__device__ __host__ static mat3 rotation_dy(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			-sinTheta,	0.0,	cosTheta,
			0.0,		1.0,	0.0,
			-cosTheta,	0.0,	-sinTheta
			);
	}

	__device__ __host__ static mat3 rotation_dz(float theta) {
		float cosTheta = cos(theta), sinTheta = sin(theta);
		return mat3(
			-sinTheta,	-cosTheta,	0.0,
			cosTheta,	-sinTheta,	0.0,
			0.0,		0.0,		1.0
			);
	}

	__device__ __host__ static mat3 rotation(float rx, float ry, float rz) {
		mat3 mat = rotation_z(rz);
		mat *= rotation_y(ry);
		mat *= rotation_x(rx);
		return mat;
	}

	__device__ __host__ static void jacobian(
		float rx, float ry, float rz,
		mat3& Jx, mat3& Jy, mat3& Jz
		) {
			mat3 Rx = rotation_x(rx);
			mat3 Ry = rotation_y(ry);
			mat3 Rz = rotation_z(rz);

			Jx = Rz * Ry * rotation_dx(rx);
			Jy = Rz * rotation_dy(ry) * Rx;
			Jz = rotation_dz(rz) * Ry * Rx;
	}

	__device__ __host__ mat3 operator-() const {
		return mat3(-elem[0], -elem[1], -elem[2], -elem[3], -elem[4], -elem[5], 
			-elem[6], -elem[7], -elem[8]);
	}

	__device__ __host__ mat3 operator+(const mat3& m) const {
		return mat3(
			elem[0]+m(0), elem[1]+m(1), elem[2]+m(2),
			elem[3]+m(3), elem[4]+m(4), elem[5]+m(5),
			elem[6]+m(6), elem[7]+m(7), elem[8]+m(8)
			);
	}

	__device__ __host__ mat3 operator-(const mat3& m) {
		return mat3(
			elem[0]-m(0), elem[1]-m(1), elem[2]-m(2),
			elem[3]-m(3), elem[4]-m(4), elem[5]-m(5),
			elem[6]-m(6), elem[7]-m(7), elem[8]-m(8)
			);
	}

	__device__ __host__ mat3 operator*(const mat3& m) {
		mat3 res;
		res(0) = elem[0] * m(0) + elem[1] * m(3) + elem[2] * m(6);
		res(1) = elem[0] * m(1) + elem[1] * m(4) + elem[2] * m(7);
		res(2) = elem[0] * m(2) + elem[1] * m(5) + elem[2] * m(8);

		res(3) = elem[0] * m(0) + elem[1] * m(3) + elem[2] * m(6);
		res(4) = elem[0] * m(1) + elem[1] * m(4) + elem[2] * m(7);
		res(5) = elem[0] * m(2) + elem[1] * m(5) + elem[2] * m(8);

		res(6) = elem[0] * m(0) + elem[1] * m(3) + elem[2] * m(6);
		res(7) = elem[0] * m(1) + elem[1] * m(4) + elem[2] * m(7);
		res(8) = elem[0] * m(2) + elem[1] * m(5) + elem[2] * m(8);

		return res;
	}

	__device__ __host__ mat3 operator*=(const mat3& m) {
		(*this) = (*this)*m;
		return (*this);
	}

	__device__ __host__ float3 operator*(const float3& v) {
		return make_float3(
			elem[0] * v.x + elem[1] * v.y + elem[2] * v.z,
			elem[3] * v.x + elem[4] * v.y + elem[5] * v.z,
			elem[6] * v.x + elem[7] * v.y + elem[8] * v.z
			);
	}

	__device__ __host__ mat3 operator*(float f) {
		return mat3(
			elem[0] * f, elem[1] * f, elem[2] * f,
			elem[3] * f, elem[4] * f, elem[5] * f,
			elem[6] * f, elem[7] * f, elem[8] * f
			);
	}

	__device__ __host__ friend mat3 operator*(float f, const mat3& m);

	__device__ __host__ mat3 trans() const {
		return mat3(
			elem[0], elem[3], elem[6],
			elem[1], elem[4], elem[7],
			elem[2], elem[5], elem[8]
		);
	}

	__device__ __host__ float det() const {
		return elem[0]*(elem[4]*elem[8]-elem[5]*elem[7])
			-elem[1]*(elem[3]*elem[8]-elem[5]*elem[6])
			+elem[2]*(elem[3]*elem[7]-elem[4]*elem[6]);
	}

	__device__ __host__ mat3 inv() const {
		float D = det();
		if( D == 0 ) return mat3::zero();
		else {
			float invD = 1.0f / D;
			mat3 res;
			res(0, 0) = (elem[4] * elem[8] - elem[7] * elem[5]) * invD;
			res(0, 1) = (elem[7] * elem[2] - elem[1] * elem[8]) * invD;
			res(0, 2) = (elem[1] * elem[5] - elem[4] * elem[2]) * invD;

			res(1, 0) = (elem[5] * elem[6] - elem[3] * elem[8]) * invD;
			res(1, 1) = (elem[8] * elem[0] - elem[6] * elem[2]) * invD;
			res(1, 2) = (elem[2] * elem[3] - elem[0] * elem[5]) * invD;

			res(2, 0) = (elem[3] * elem[7] - elem[4] * elem[6]) * invD;
			res(2, 1) = (elem[6] * elem[1] - elem[7] * elem[0]) * invD;
			res(2, 2) = (elem[0] * elem[4] - elem[1] * elem[3]) * invD;

			return res;
		}
	}

	__device__ __host__ float operator()(int idx) const { return elem[idx]; }
	__device__ __host__ float& operator()(int idx) { return elem[idx]; }

	__device__ __host__ float operator()(int r, int c) const { return elem[r*3+c]; }
	__device__ __host__ float& operator()(int r, int c) { return elem[r*3+c]; }

	float elem[9];
};

__device__ __host__ __inline__ mat3 operator*(float f, const mat3& m) {
	return mat3(
		m.elem[0] * f, m.elem[1] * f, m.elem[2] * f,
		m.elem[3] * f, m.elem[4] * f, m.elem[5] * f,
		m.elem[6] * f, m.elem[7] * f, m.elem[8] * f
		);
}

class mat4 {
public:
	__device__ __host__ mat4(){
		for(int i=0;i<16;i++) elem[i] = 0;
	}
	__device__ __host__ mat4(const mat4& m){
		for(int i=0;i<16;i++) elem[i] = m.elem[i];
	}
	__device__ __host__ mat4(
		float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33
		)
	{
		elem[ 0] = m00; elem[ 1] = m01; elem[ 2] = m02; elem[ 3] = m03;
		elem[ 4] = m10; elem[ 5] = m11; elem[ 6] = m12; elem[ 7] = m13;
		elem[ 8] = m20; elem[ 9] = m21; elem[10] = m22; elem[11] = m23;
		elem[12] = m30; elem[13] = m31; elem[14] = m32; elem[15] = m33;
	}
	__device__ __host__ mat4(float *m) {
		elem[0] = m[0]; elem[1] = m[1]; elem[2] = m[2]; elem[3] = m[3];
		elem[4] = m[4]; elem[5] = m[5];	elem[6] = m[6]; elem[7] = m[7]; 
		elem[8] = m[8]; elem[9] = m[9]; elem[10] = m[10]; elem[11] = m[11];
		elem[12] = m[12]; elem[13] = m[13]; elem[14] = m[14]; elem[15] = m[15];
	}
	__device__ __host__ ~mat4(){}

	__device__ __host__ mat4& operator=(const mat4& m){
		for(int i=0;i<16;i++) elem[i] = m.elem[i];
		return (*this);
	}

	__device__ __host__ static mat4 zero(){ return mat4(); }
	__device__ __host__ static mat4 identity(){ 
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
			); 
	}

	__device__ __host__ mat4 trans() const {
		return mat4(
			elem[0], elem[4], elem[8], elem[12],
			elem[1], elem[5], elem[9], elem[13],
			elem[2], elem[6], elem[10], elem[14],
			elem[3], elem[7], elem[11], elem[15]
		);
	}

	__device__ __host__ mat4 inv() const {
		const float* pm = &(elem[0]);
		float S0 = pm[0] * pm[5] - pm[1] * pm[4];
		float S1 = pm[0] * pm[6] - pm[2] * pm[4];
		float S2 = pm[0] * pm[7] - pm[3] * pm[4];
		float S3 = pm[1] * pm[6] - pm[2] * pm[5];
		float S4 = pm[1] * pm[7] - pm[3] * pm[5];
		float S5 = pm[2] * pm[7] - pm[3] * pm[6];

		float C5 = pm[10] * pm[15] - pm[11] * pm[14];
		float C4 = pm[9] * pm[15] - pm[11] * pm[13];
		float C3 = pm[9] * pm[14] - pm[10] * pm[13];
		float C2 = pm[8] * pm[15] - pm[11] * pm[12];
		float C1 = pm[8] * pm[14] - pm[10] * pm[12];
		float C0 = pm[8] * pm[13] - pm[9] * pm[12];

		// If determinant equals 0, there is no inverse
		float D = S0 * C5 - S1 * C4 + S2 * C3 + S3 * C2 - S4 * C1 + S5 * C0;
		if(fabs(D) <= 1e-8) return mat4();

		float dinv = 1.0f / D;
		// Compute adjugate matrix
		mat4 mat(
			pm[5] * C5  - pm[6] * C4  + pm[7] * C3,  -pm[1] * C5 + pm[2] * C4  - pm[3] * C3,
			pm[13] * S5 - pm[14] * S4 + pm[15] * S3, -pm[9] * S5 + pm[10] * S4 - pm[11] * S3,

			-pm[4] * C5  + pm[6] * C2  - pm[7] * C1,   pm[0] * C5 - pm[2] * C2  + pm[3] * C1,
			-pm[12] * S5 + pm[14] * S2 - pm[15] * S1,  pm[8] * S5 - pm[10] * S2 + pm[11] * S1,

			pm[4] * C4  - pm[5] * C2  + pm[7] * C0,  -pm[0] * C4 + pm[1] * C2  - pm[3] * C0,
			pm[12] * S4 - pm[13] * S2 + pm[15] * S0, -pm[8] * S4 + pm[9] * S2  - pm[11] * S0,

			-pm[4] * C3  + pm[5] * C1  - pm[6] * C0,   pm[0] * C3 - pm[1] * C1  + pm[2] * C0,
			-pm[12] * S3 + pm[13] * S1 - pm[14] * S0,  pm[8] * S3 - pm[9] * S1  + pm[10] * S0 
			);

		mat *= dinv;

		return mat;
	}

	__device__ __host__ float det() const {
		const float* pm = &(elem[0]);
		float S0 = pm[0] * pm[5] - pm[1] * pm[4];
		float S1 = pm[0] * pm[6] - pm[2] * pm[4];
		float S2 = pm[0] * pm[7] - pm[3] * pm[4];
		float S3 = pm[1] * pm[6] - pm[2] * pm[5];
		float S4 = pm[1] * pm[7] - pm[3] * pm[5];
		float S5 = pm[2] * pm[7] - pm[3] * pm[6];

		float C5 = pm[10] * pm[15] - pm[11] * pm[14];
		float C4 = pm[9] * pm[15] - pm[11] * pm[13];
		float C3 = pm[9] * pm[14] - pm[10] * pm[13];
		float C2 = pm[8] * pm[15] - pm[11] * pm[12];
		float C1 = pm[8] * pm[14] - pm[10] * pm[12];
		float C0 = pm[8] * pm[13] - pm[9] * pm[12];

		return S0 * C5 - S1 * C4 + S2 * C3 + S3 * C2 - S4 * C1 + S5 * C0;
	}

	__device__ __host__ mat4 operator+(const mat4& m) {
		return mat4(
			elem[0] + m.elem[0], elem[1] + m.elem[1], elem[2] + m.elem[2], elem[3] + m.elem[3],
			elem[4] + m.elem[4], elem[5] + m.elem[5], elem[6] + m.elem[6], elem[7] + m.elem[7],
			elem[8] + m.elem[8], elem[9] + m.elem[9], elem[10] + m.elem[10], elem[11] + m.elem[11],
			elem[12] + m.elem[12], elem[13] + m.elem[13], elem[14] + m.elem[14], elem[15] + m.elem[15]
		);
	}
	__device__ __host__ mat4 operator-(const mat4& m) {
		return mat4(
			elem[0] - m.elem[0], elem[1] - m.elem[1], elem[2] - m.elem[2], elem[3] - m.elem[3],
			elem[4] - m.elem[4], elem[5] - m.elem[5], elem[6] - m.elem[6], elem[7] + m.elem[7],
			elem[8] - m.elem[8], elem[9] - m.elem[9], elem[10] - m.elem[10], elem[11] - m.elem[11],
			elem[12] - m.elem[12], elem[13] - m.elem[13], elem[14] - m.elem[14], elem[15] - m.elem[15]
		);
	}
	__device__ __host__ mat4 operator*(const mat4& m) {
		mat4 res;
		res(0)  = elem[0] * m(0) + elem[1] * m(4) + elem[2] * m( 8) + elem[3] * m(12);
		res(1)  = elem[0] * m(1) + elem[1] * m(5) + elem[2] * m( 9) + elem[3] * m(13);
		res(2)  = elem[0] * m(2) + elem[1] * m(6) + elem[2] * m(10) + elem[3] * m(14);
		res(3)  = elem[0] * m(3) + elem[1] * m(7) + elem[2] * m(11) + elem[3] * m(15);

		res(4)  = elem[4] * m(0, 0) + elem[5] * m(1, 0) + elem[6] * m(2, 0) + elem[7] * m(3, 0);
		res(5)  = elem[4] * m(0, 1) + elem[5] * m(1, 1) + elem[6] * m(2, 1) + elem[7] * m(3, 1);
		res(6)  = elem[4] * m(0, 2) + elem[5] * m(1, 2) + elem[6] * m(2, 2) + elem[7] * m(3, 2);
		res(7)  = elem[4] * m(0, 3) + elem[5] * m(1, 3) + elem[6] * m(2, 3) + elem[7] * m(3, 3);

		res(8)  = elem[8] * m(0, 0) + elem[9] * m(1, 0) + elem[10] * m(2, 0) + elem[11] * m(3, 0);
		res(9)  = elem[8] * m(0, 1) + elem[9] * m(1, 1) + elem[10] * m(2, 1) + elem[11] * m(3, 1);
		res(10) = elem[8] * m(0, 2) + elem[9] * m(1, 2) + elem[10] * m(2, 2) + elem[11] * m(3, 2);
		res(11) = elem[8] * m(0, 3) + elem[9] * m(1, 3) + elem[10] * m(2, 3) + elem[11] * m(3, 3);

		res(12) = elem[12] * m(0, 0) + elem[13] * m(1, 0) + elem[14] * m(2, 0) + elem[15] * m(3, 0);
		res(13) = elem[12] * m(0, 1) + elem[13] * m(1, 1) + elem[14] * m(2, 1) + elem[15] * m(3, 1);
		res(14) = elem[12] * m(0, 2) + elem[13] * m(1, 2) + elem[14] * m(2, 2) + elem[15] * m(3, 2);
		res(15) = elem[12] * m(0, 3) + elem[13] * m(1, 3) + elem[14] * m(2, 3) + elem[15] * m(3, 3);
		return res;
	}

	__device__ __host__ float4 operator*(const float4& v) {
		float4 res;
		res.x = v.x * elem[0] + v.y * elem[1] + v.z * elem[2] + v.w * elem[3];
		res.y = v.x * elem[4] + v.y * elem[5] + v.z * elem[6] + v.w * elem[7];
		res.z = v.x * elem[8] + v.y * elem[9] + v.z * elem[10] + v.w * elem[11];
		res.w = v.x * elem[12] + v.y * elem[13] + v.z * elem[14] + v.w * elem[15];
		return res;
	}

	__device__ __host__ float3 operator*(const float3& v) {
		float4 vv = make_float4(v, 1.0);
		vv = (*this) * vv;
		return make_float3(vv.x, vv.y, vv.z);
	}

	__device__ __host__ mat4 operator*(float f) {
		return mat4(
			elem[0]*f, elem[1]*f, elem[2]*f, elem[3]*f,
			elem[4]*f, elem[5]*f, elem[6]*f, elem[7]*f,
			elem[8]*f, elem[9]*f, elem[10]*f, elem[11]*f,
			elem[12]*f, elem[13]*f, elem[14]*f, elem[15]*f
			);
	}
	__device__ __host__ mat4 operator+=(const mat4& m) {
		elem[0] += m.elem[0]; elem[1] += m.elem[1]; elem[2] += m.elem[2]; elem[3] += m.elem[3];
		elem[4] += m.elem[4]; elem[5] += m.elem[5]; elem[6] += m.elem[6]; elem[7] += m.elem[7];
		elem[8] += m.elem[8]; elem[9] += m.elem[9]; elem[10] += m.elem[10]; elem[11] += m.elem[11];
		elem[12] += m.elem[12]; elem[13] += m.elem[13]; elem[14] += m.elem[14]; elem[15] += m.elem[15];
		return (*this);
	}
	__device__ __host__ mat4 operator-=(const mat4& m) {
		elem[0] -= m.elem[0]; elem[1] -= m.elem[1]; elem[2] -= m.elem[2]; elem[3] -= m.elem[3];
		elem[4] -= m.elem[4]; elem[5] -= m.elem[5]; elem[6] -= m.elem[6]; elem[7] -= m.elem[7];
		elem[8] -= m.elem[8]; elem[9] -= m.elem[9]; elem[10] -= m.elem[10]; elem[11] -= m.elem[11];
		elem[12] -= m.elem[12]; elem[13] -= m.elem[13]; elem[14] -= m.elem[14]; elem[15] -= m.elem[15];
		return (*this);
	}

	__device__ __host__ mat4 operator*=(const mat4& m) {
		mat4 res;
		res(0)  = elem[0] * m(0) + elem[1] * m(4) + elem[2] * m( 8) + elem[3] * m(12);
		res(1)  = elem[0] * m(1) + elem[1] * m(5) + elem[2] * m( 9) + elem[3] * m(13);
		res(2)  = elem[0] * m(2) + elem[1] * m(6) + elem[2] * m(10) + elem[3] * m(14);
		res(3)  = elem[0] * m(3) + elem[1] * m(7) + elem[2] * m(11) + elem[3] * m(15);

		res(4)  = elem[4] * m(0, 0) + elem[5] * m(1, 0) + elem[6] * m(2, 0) + elem[7] * m(3, 0);
		res(5)  = elem[4] * m(0, 1) + elem[5] * m(1, 1) + elem[6] * m(2, 1) + elem[7] * m(3, 1);
		res(6)  = elem[4] * m(0, 2) + elem[5] * m(1, 2) + elem[6] * m(2, 2) + elem[7] * m(3, 2);
		res(7)  = elem[4] * m(0, 3) + elem[5] * m(1, 3) + elem[6] * m(2, 3) + elem[7] * m(3, 3);

		res(8)  = elem[8] * m(0, 0) + elem[9] * m(1, 0) + elem[10] * m(2, 0) + elem[11] * m(3, 0);
		res(9)  = elem[8] * m(0, 1) + elem[9] * m(1, 1) + elem[10] * m(2, 1) + elem[11] * m(3, 1);
		res(10) = elem[8] * m(0, 2) + elem[9] * m(1, 2) + elem[10] * m(2, 2) + elem[11] * m(3, 2);
		res(11) = elem[8] * m(0, 3) + elem[9] * m(1, 3) + elem[10] * m(2, 3) + elem[11] * m(3, 3);

		res(12) = elem[12] * m(0, 0) + elem[13] * m(1, 0) + elem[14] * m(2, 0) + elem[15] * m(3, 0);
		res(13) = elem[12] * m(0, 1) + elem[13] * m(1, 1) + elem[14] * m(2, 1) + elem[15] * m(3, 1);
		res(14) = elem[12] * m(0, 2) + elem[13] * m(1, 2) + elem[14] * m(2, 2) + elem[15] * m(3, 2);
		res(15) = elem[12] * m(0, 3) + elem[13] * m(1, 3) + elem[14] * m(2, 3) + elem[15] * m(3, 3);
		(*this) = res;
		return (*this);
	}

	__device__ __host__ mat4 operator*=(float f) {
		elem[0] *= f; elem[1] *= f; elem[2] *= f; elem[3] *= f;
		elem[4] *= f; elem[5] *= f; elem[6] *= f; elem[7] *= f;
		elem[8] *= f; elem[9] *= f; elem[10] *= f; elem[11] *= f;
		elem[12] *= f; elem[13] *= f; elem[14] *= f; elem[15] *= f;
		return (*this);
	}

	__device__ __host__ float operator()(int idx) const { return elem[idx]; }
	__device__ __host__ float& operator()(int idx) { return elem[idx]; }

	__device__ __host__ float operator()(int r, int c) const { return elem[r*4+c]; }
	__device__ __host__ float& operator()(int r, int c) { return elem[r*4+c]; }

	float elem[16];
};