#include "UniformGrid.h"

__device__ bool IsRayBoxIntersect(Vec3 minPt, Vec3 maxPt, Ray ray, Vec3* intersectPoint);
__device__ bool IsRayTriangleIntersect(Vec3* triangleVertices, Ray ray, Vec3 hitNormal, Vec3* intersectionPoint, float* distance);
__device__ bool IsRayPlaneIntersect(float A, float B, float C, float D, Ray ray, Vec3* intersectionPoint, Vec3* hitNormal, double* distance);
__device__ bool IsRaySphereIntersect(Vec3 centerS, double radius, Ray ray, Vec3* intersectionPoint, Vec3* hitNormal, double* distance);


__device__ bool IsRayPlaneIntersect(	float A,
									float B,
									float C,
									float D,
									Ray ray,
									Vec3* intersectionPoint,
									Vec3* hitNormal,
									float* distance)
{
	Vec3 pNormal = Vector3(A, B, C); //Compute plane normal
	pNormal = ScalarDiv(pNormal, Magnitude(pNormal));
	double t;

	double denominator = DotProduct(pNormal, ray.dir);
	double numerator = D + DotProduct(pNormal, ray.orig);

	if (denominator != 0)
	{
		t = -1 * (numerator / denominator); //Parametric variable

		if (t > 0.0)
		{
			*intersectionPoint = ScalarMul(ray.dir, t);
			*intersectionPoint = Add(ray.orig, *intersectionPoint);
			*hitNormal = pNormal;
			*distance = sqrt((ray.orig.x - intersectionPoint->x) * (ray.orig.x - intersectionPoint->x) +
				(ray.orig.y - intersectionPoint->y) * (ray.orig.y - intersectionPoint->y) +
				(ray.orig.z - intersectionPoint->z) * (ray.orig.z - intersectionPoint->z));
			return true;
		}
		else
		{
			*intersectionPoint = Vector3(0, 0, 0);//No intersection point found
		}
	}

	return false;//No intersection point found
}


/*
Function: Checks for the intersection point of the ray and the sphere by a method given in ScratchAPixel.
Parameters: Center(Sphere), Radius (Sphere), instance of Ray struct.
Output: Returns the whether the ray intersects with Sphere.
Out Param: intersection Point, Distance from the sphere.
*/


__device__ bool IsRaySphereIntersect(Vec3 centerS,
									float radius,
									Ray ray,
									Vec3* intersectionPoint,
									Vec3* hitNormal,
									float* distance)
{
	double t0, t1;//parametric constants
	bool isIntersect = true;
	double radius2 = radius * radius;
	Vec3 L = Sub(ray.orig, centerS);
	double tca = DotProduct(L, ray.dir);

	if (tca < 0) isIntersect = false;

	double d2 = DotProduct(L, L) - tca * tca;

	if (d2 > radius2) isIntersect = false;
	double thc = sqrt(radius2 - d2);

	t0 = tca - thc;
	t1 = tca + thc;

	if (t0 > t1)
	{
		double temp = t0;
		t0 = t1;
		t1 = temp;
	}

	if (t0 < 0)
	{
		t0 = t1; // if t0 is negative, let's use t1 instead 
		if (t0 < 0)
			isIntersect = false; // both t0 and t1 are negative 
	}

	if (isIntersect)
	{
		*intersectionPoint = ScalarMul(ray.dir, t0);
		*intersectionPoint = Add(ray.orig, *intersectionPoint);
		*hitNormal = Normalize(Sub(centerS, *intersectionPoint));
		*distance = sqrt((ray.orig.x - intersectionPoint->x) * (ray.orig.x - intersectionPoint->x) +
			(ray.orig.y - intersectionPoint->y) * (ray.orig.y - intersectionPoint->y) +
			(ray.orig.z - intersectionPoint->z) * (ray.orig.z - intersectionPoint->z));

		return true;
	}
	else
	{
		*intersectionPoint = Vector3(0, 0, 0);//No intersection point found
		return false;
	}
}

__device__ bool IsRayTriangleIntersect(Vec3* triangleVertices, Ray ray, Vec3 hitNormal, Vec3* intersectionPoint, float* distance)
{
	double kEpsilon = 0.000001;

	double u, v, t;

	Vec3 v0v1 = Sub(triangleVertices[0], triangleVertices[1]);
	Vec3 v0v2 = Sub(triangleVertices[0], triangleVertices[2]);
	
	Vec3 pvec = CrossProduct(ray.dir, v0v2);
	double det = DotProduct(v0v1, pvec);

	if (det < kEpsilon) return false;

	if (fabs((float)det) < kEpsilon) return false;

	double invDet = 1 / det;

	Vec3 tvec = Sub(triangleVertices[0], ray.orig);

	u = DotProduct(tvec, pvec) * invDet;

	if (u < 0 || u > 1) return false;

	Vec3 qvec = CrossProduct(tvec, v0v1);
	v = DotProduct(ray.dir, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	t = DotProduct(v0v2, qvec) * invDet;

	if (t > 0.0)
	{

		*intersectionPoint = ScalarMul(ray.dir, t);
		*intersectionPoint = Add(ray.orig, *intersectionPoint);
		*distance = Magnitude(Sub(*intersectionPoint, ray.orig));

		return true;

	}
	
	return false;
}


__device__ bool IsRayBoxIntersect(Vec3 minPt, Vec3 maxPt, Ray ray, Vec3* intersectPoint)
{
	if (ray.orig.x >= minPt.x && ray.orig.x <= maxPt.x &&
		ray.orig.y >= minPt.y && ray.orig.y <= maxPt.y &&
		ray.orig.z >= minPt.z && ray.orig.z <= maxPt.z)
	{
		*intersectPoint = ray.orig;
		return true;
	}

	double tmin = (minPt.x - ray.orig.x) / ray.dir.x;
	double tmax = (maxPt.x - ray.orig.x) / ray.dir.x;

	if (tmin > tmax)
	{
		double temp = tmin;
		tmin = tmax;
		tmax = temp;
	}

	double tymin = (minPt.y - ray.orig.y) / ray.dir.y;
	double tymax = (maxPt.y - ray.orig.y) / ray.dir.y;

	if (tymin > tymax)
	{
		double temp = tymin;
		tymin = tymax;
		tymax = temp;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	double tzmin = (minPt.z - ray.orig.z) / ray.dir.z;
	double tzmax = (maxPt.z - ray.orig.z) / ray.dir.z;

	if (tzmin > tzmax)
	{
		double temp = tzmin;
		tzmin = tzmax;
		tzmax = temp;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	*intersectPoint = ScalarMul(ray.dir, tmin);
	*intersectPoint = Add(ray.orig, *intersectPoint);
	return true;

}
