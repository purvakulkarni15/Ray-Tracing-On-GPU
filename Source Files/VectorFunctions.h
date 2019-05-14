#include "decl.h"

__host__ __device__ Vec3 Vector3(double x, double y, double z)
{
	Vec3 v;
	v.x = x;
	v.y = y;
	v.z = z;

	return v;
}

__host__ __device__ float DotProduct(Vec3 v1, Vec3 v2)
{
	return(((v1.x*v2.x)+(v1.y*v2.y)+(v1.z*v2.z)));
}

__host__ __device__ Vec3 CrossProduct(Vec3 v1, Vec3 v2)
{
	Vec3 vRet;
	vRet.x = (v1.y*v2.z - v1.z*v2.y);
	vRet.y = (v1.z*v2.x - v1.x*v2.z);
	vRet.z = (v1.x*v2.y - v1.y*v2.x);

	return vRet;
}

__host__ __device__ float Magnitude(Vec3 v)
{
	float sqr = v.x*v.x + v.y*v.y + v.z*v.z;
	return((float)sqrt((double)sqr));
}

__host__ __device__ Vec3 Add(Vec3 v1, Vec3 v2)
{
	Vec3 vRet;
	vRet.x = v1.x + v2.x;
	vRet.y = v1.y + v2.y;
	vRet.z = v1.z + v2.z;
	
	return vRet;
}

__host__ __device__ Vec3 Sub(Vec3 v1, Vec3 v2)
{
	Vec3 vRet;
	vRet.x = v2.x - v1.x;
	vRet.y = v2.y - v1.y;
	vRet.z = v2.z - v1.z;
	
	return vRet;
}

__host__ __device__ Vec3 ScalarMul(Vec3 v, float mul)
{
	Vec3 vRet;
	vRet.x = v.x*mul;
	vRet.y = v.y*mul;
	vRet.z = v.z*mul;

	return vRet;
}

__host__ __device__ Vec3 ScalarDiv(Vec3 v, float div)
{
	Vec3 vRet;
	vRet.x = v.x/div;
	vRet.y = v.y/div;
	vRet.z = v.z/div;

	return vRet;
}

__host__ __device__ Vec3 Normalize(Vec3 v)
{
	return(ScalarDiv(v, Magnitude(v)));
}

__host__ __device__ Vec3 reflect(Vec3 I, Vec3 N)
{
	return Sub(ScalarMul(N, 2 * DotProduct(I, N)), I);
}

__host__ __device__ Vec3 Scale(Vec3 v, float xScale, float yScale, float zScale)
{
	Vec3 vRet;
	vRet.x = v.x * xScale;
	vRet.y = v.y * yScale;
	vRet.z = v.z * zScale;

	return vRet;
}

__host__ __device__ Vec3 Translate(Vec3 v, float xTrans, float yTrans, float zTrans)
{
	Vec3 vRet;
	vRet.x = v.x + xTrans;
	vRet.y = v.y + yTrans;
	vRet.z = v.z + zTrans;

	return vRet;
}
