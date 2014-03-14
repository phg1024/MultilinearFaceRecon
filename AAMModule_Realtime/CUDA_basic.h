#ifndef GPUBASIC
#define GPUBASIC

#include <crtdbg.h>
#include <cublas_v2.h>
#include <cula.h>
#include <fstream>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdio>



using namespace std;
using namespace thrust;

template <typename T>
void printArray(T* A, int N, ostream& os = cout)
{
	for(int i=0;i<N;i++)
		os << A[i] << ' ';
	os << endl;
}

template <typename T>
void print2DArray(T** A, int rows, int cols, ostream& os = cout)
{
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			os << A[i][j] << ' ';
		}
		os << endl;
	}
}

template <typename T>
void print2DArray(T* A, int rows, int cols, ostream& os = cout)
{
	for(int i=0, idx=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			os << A[idx++] << ' ';
		}
		os << endl;
	}
}

template <typename T> 
__host__ inline void writeback(T* ptr, int size, const string& filename) {
	vector<T> v(size);
	cudaMemcpy(&(v[0]), ptr, sizeof(T)*size, cudaMemcpyDeviceToHost);
	ofstream fout(filename);
	if( !fout ) {
		cerr << "failed to write file " << filename << endl;
		return;
	}
	printArray(&(v[0]), size, fout);
	fout.close();
}

template <typename T> 
__host__ inline void writeback(T* ptr, int rows, int cols, const string& filename) {
	vector<T> v(rows*cols);
	cudaMemcpy(&(v[0]), ptr, sizeof(T)*rows*cols, cudaMemcpyDeviceToHost);
	ofstream fout(filename);
	if( !fout ) {
		cerr << "failed to write file " << filename << endl;
		return;
	}
	print2DArray<T>(&v[0], rows, cols, fout);
	fout.close();
}

#ifdef _DEBUG
#define CHECKCUDASTATE_DEBUG
#endif

__host__ inline void checkCudaState_impl(const char* file = __FILE__ , int line = __LINE__ ) {
	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)  {
		fprintf(stderr, "CUDA error at %s: line# %d\t%s\n",file, line, cudaGetErrorString(err));
		::system("pause");
		//exit(-1);
	}
}

#ifdef CHECKCUDASTATE_DEBUG
#define checkCudaState() checkCudaState_impl(__FILE__, __LINE__)
#else
#define checkCudaState()
#endif

#ifdef _DEBUG
#define CUDA_CALL( call ) do { \
	cudaError_t err; \
	err = (call); \
	if( err != cudaSuccess ) \
	{ \
		fprintf( stderr, "error in CUDA call in file '%s', line: %d\n" \
					"error %d: %s\n", \
							__FILE__, __LINE__, \
					err, cudaGetErrorString( err ) ); \
		if(::_CrtDbgReport(_CRT_ASSERT, __FILE__, __LINE__, NULL, NULL)==1) \
		{ \
			::_CrtDbgBreak(); \
		} \
	} \
} while(0)

#define CUBLAS_CALL( call ) do { \
	cublasStatus_t err; \
	err = (call); \
	if( err != CUBLAS_STATUS_SUCCESS ) \
	{ \
	fprintf( stderr, "error in CUBLAS call in file '%s', line: %d\n" \
	"error %d:\n", \
	__FILE__, __LINE__, \
	err); \
	if(::_CrtDbgReport(_CRT_ASSERT, __FILE__, __LINE__, NULL, NULL)==1) \
		{ \
		::_CrtDbgBreak(); \
		} \
	} \
} while(0)

#define CULA_CALL( call ) do { \
	culaStatus err; \
	culaInfo info; \
	char err_string[1000]; \
	err = (call); \
	if( err != culaNoError ) \
	{ \
	info = culaGetErrorInfo(); \
	culaGetErrorInfoString(err, info, err_string, 1000); \
	fprintf( stderr, "error in CULA call in file '%s', line: %d\n" \
	"error %d: %s, %s\n", \
	__FILE__, __LINE__, \
	err, culaGetStatusString( err ), err_string ); \
	if(::_CrtDbgReport(_CRT_ASSERT, __FILE__, __LINE__, NULL, NULL)==1) \
		{ \
		::_CrtDbgBreak(); \
		} \
	} \
} while(0)

#define CUDA_ASSERT( call ) do { \
	if(!(call))  \
	{ \
	printf("assertion in CUDA call in file '%s', line: %d\n" \
	__FILE__, __LINE__); \
	return; }\
} while(0)

#else		// FOR RELEASE
/*
 * CUDA error handling macro
 */
#define CUDA_CALL( call ) \
        { \
            cudaError_t err; \
            err = (call); \
            if( err != cudaSuccess ) \
            { \
                fprintf( stderr, "error in CUDA call in file '%s', line: %d\n" \
                        "%s\nerror %d: %s\nterminating!\n", \
                        __FILE__, __LINE__, #call, \
                        err, cudaGetErrorString( err ) ); \
				::system("pause"); \
                exit( ~0 ); \
            } \
        }

static const char *cublasGetErrorString(cublasStatus_t error)
{
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

#define CUBLAS_CALL( call ) \
		{ \
		cublasStatus_t err; \
		err = (call); \
		if( err != CUBLAS_STATUS_SUCCESS ) \
			{ \
			fprintf( stderr, "error in CUBLAS call in file '%s', line: %d\n" \
			"%s\nerror %s: %s\nterminating!\n", \
			__FILE__, __LINE__, #call, \
			cublasGetErrorString(err) ); \
			::system("pause"); \
			exit(~0); \
			} \
		}

#define CULA_CALL( call ) \
		{ \
		culaStatus err; \
		culaInfo info; \
		char err_string[1000]; \
		char *err_type; \
		err = (call); \
		if( err != culaNoError ) \
			{ \
			info = culaGetErrorInfo(); \
			err_type = "CULA Error"; \
			if(err == culaArgumentError) \
			{ \
				err_type = "Argument error"; \
			} \
			else \
			if(err == culaDataError) \
			{ \
				err_type = "Data error"; \
			} \
			culaGetErrorInfoString(err, info, err_string, 1000); \
			fprintf( stderr, "%s in CULA call in file '%s', line: %d\n" \
			"%s\nerror %d: %s, %s\nterminating!\n", \
			err_type, __FILE__, __LINE__, #call, \
			err, culaGetStatusString( err ), err_string); \
			::system("pause"); \
			exit( ~0 ); \
			} \
		}

#define CUDA_ASSERT( call )  

#define CUDA_RASSERT( call )  do { \
	if(!(call))  \
	{ \
	printf("assertion in CUDA call\n"); \
	return; }\
} while(0)

#endif

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);            \
	return EXIT_FAILURE;}} while(0)

// let *destDevice = src, where *destDevice locates in GPU
template<typename T>
void copyVarToDevice(T *destDevice, const T& src);
// return *valInDevice in host
template<typename T>
T getVarFromDevice(T *valInDevice);

// let *destDevice = src, where *destDevice locates in GPU
template<typename T>
void copyVarToDevice(T *destDevice, const T& src)
{
	CUDA_CALL(cudaMemcpy(destDevice, &src, sizeof(T), cudaMemcpyHostToDevice));
}

// return *valInDevice in host
template<typename T>
T getVarFromDevice(T *valInDevice)
{
	T val;
	CUDA_CALL(cudaMemcpy(&val, valInDevice, sizeof(T), cudaMemcpyDeviceToHost));
	return val;
}

template<typename T>
const T getVarFromDevice(const T *valInDevice)
{
	T val;
	CUDA_CALL(cudaMemcpy(&val, valInDevice, sizeof(T), cudaMemcpyDeviceToHost));
	return val;
}

template<typename T>
void NewArrayInDevice(T **ptr, int numElement)
{
	CUDA_CALL(cudaMalloc(ptr, sizeof(T) * numElement));
}

template<typename T>
T* NewArrayInDevice(int numElement, const T* srcArray = NULL)
{
	T* ret;
	CUDA_CALL(cudaMalloc(&ret, sizeof(T) * numElement));
	if(srcArray)
	{
		CUDA_CALL( cudaMemcpy(ret, srcArray, sizeof(T)*numElement, cudaMemcpyHostToDevice) );
	}
	return ret;
}

template<typename T>
void deleteArrayInDevice(T *ptr)
{
	CUDA_CALL(cudaFree(ptr));
}

template<typename T>
void copyArrayToHost(T* dstArray, const T* srcArrayGPU, int copyElem)
{
	CUDA_CALL(cudaMemcpy(dstArray, srcArrayGPU, sizeof(T)*copyElem, cudaMemcpyDeviceToHost));
}

template<typename T>
void copyArrayToDevice(T* dstArray, const T* srcArrayGPU, int copyELem)
{
	CUDA_CALL(cudaMemcpy(dstArray, srcArrayGPU, sizeof(T)*copyELem, cudaMemcpyHostToDevice));
}

template<typename T>
T* getArrayFromDevice(const T* srcArrayGPU, int copyELem)
{
	T* ret = new T[copyELem];
	CUDA_CALL(cudaMemcpy(ret, srcArrayGPU, sizeof(T)*copyELem, cudaMemcpyHostToDevice));
	return ret;
}


double* DeviceDoubleArray(int size);
double* DeviceDoubleArray(int size, const double* host_array);
double* DeviceDoubleArray(int size, double value);
double* HostDoubleArray(int size, const double* device_array);
int* DeviceIntArray(int size, const int* host_array);
int* DeviceIntArray(int size, int value);
int* HostIntArray(int size, const int* device_array);

__host__ __device__ inline bool operator==(const float4& a, const float4& b)
{
	if(a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w)
	{
		return true;
	}
	return false;
}

__host__ __device__ inline bool operator!=(const float4& a, const float4& b)
{
	if(a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w)
	{
		return true;
	}
	return false;
}

__host__ __device__ inline void printData(const char *name, const float4& f, const char *tail)
{
	printf("%s (%0.1f %0.1f %0.1f %0.1f) %s", name, f.x, f.y, f.z, f.w, tail);
}

__host__ __device__ inline float Dis2(const float4& p1, const float4& p2) 
{ 
	float px = p1.x - p2.x;
	float py = p1.y - p2.y;
	float pz = p1.z - p2.z;
	return px*px + py*py + pz*pz; 
}

__host__ __device__ inline float Dis(const float4& p1, const float4& p2) 
{ 
	float px = p1.x - p2.x;
	float py = p1.y - p2.y;
	float pz = p1.z - p2.z;
	return sqrt(px*px + py*py + pz*pz); 
}

__device__ inline float my_round(float number)
{
	return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

__device__ inline int dataClip(int data, int minVal, int maxVal)
{
	return ( (data < minVal) ? minVal : (data > maxVal) ? maxVal : data );
}

//__host__ __device__ inline float4 operator+(const float4& a, const float4& b)
//{
//	return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
//}

//__host__ __device__ inline float4 operator/(const float4& a, const float& b)
//{
//	return make_float4(a.x/b, a.y/b, a.z/b, a.w/b);
//}
inline std::ostream& operator<<(std::ostream& os, const float4& p) 
{ 
	os<<p.x<<" "<<p.y<<" "<<p.z<<" "<<p.w; 
	return os; 
}

template<typename Type>
void MemsetCuda(Type* ptr, Type value, int elementCount)
{
	CUDA_CALL( cudaMemset(ptr, value, sizeof(Type)*elementCount) );
}

inline void showCUDAMemoryUsage(const char* str = NULL)
{
	size_t free_byte ;
	size_t total_byte ;
	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
	if ( cudaSuccess != cuda_status ){
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
		exit(1);
	}
	
	double free_db = (double)free_byte ;
	double total_db = (double)total_byte ;
	double used_db = total_db - free_db ;
	if(str != NULL)
	{
		printf("%s GPU memory usage: \n\tused = %f \n\tfree = %f MB \n\ttotal = %f MB\n", str,
			used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
	} else
	{
		printf("GPU memory usage: \n\tused = %f \n\tfree = %f MB \n\ttotal = %f MB\n",
			used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
	}
}

#endif