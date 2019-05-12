#include "RayObjectIntersectionFunctions.h"

//Function Declarations
__global__ void RayTracer(Voxel* uniformGrid, Face* f, Plane* planeList, Sphere* sphereList, int planeListSize, int sphereListSize, Vec3* sensorGrid, Vec3* imageDataMat, SceneBB sceneBB, int x, int y, int xDim, int yDim, int zDim, int Nx, int Ny, int Nz);
__device__ bool TraceRay(Voxel* uniformGrid, Face* f, Plane* planeList, Sphere* sphereList, int planeListSize, int sphereListSize, Ray* ray, SceneBB sceneBB, int x, int y, int xDim, int yDim, int zDim, int Nx, int Ny, int Nz);

__device__ bool TraceRay(Voxel* uniformGrid, Face* f, Plane* planeList, Sphere* sphereList, int planeListSize, int sphereListSize, Ray* ray, SceneBB sceneBB, int x, int y, int xDim, int yDim, int zDim, int Nx, int Ny, int Nz)
{
	
	Vec3 curpos; //Ray-Bounding Box intersection Point
	Vec3 rayObjectIntersectionPoint;
	Vec3 rayObjectHitNormal;
	float rayObjectDistance;

	bool isIntersect = false;

	if (IsRayBoxIntersect(sceneBB.minP, sceneBB.maxP, *ray, &curpos))
	{
		Vec3 deltaT, tMax;
		Vec3* voxelWorldCoord;
		int* voxelGridCoord;
		int uniformGridIndex;
		int stepX = 1, stepY = 1, stepZ = 1;
		int outX = xDim, outY = yDim, outZ = zDim;

		tMax.x = INFINITY;
		tMax.y = INFINITY;
		tMax.z = INFINITY;

		deltaT.x = INFINITY;
		deltaT.y = INFINITY;
		deltaT.z = INFINITY;


		voxelGridCoord = (int*)malloc(sizeof(int) * 3);
		WorldToVoxel(curpos, voxelGridCoord, sceneBB, Nx, Ny, Nz);

		voxelWorldCoord = (Vec3*)malloc(sizeof(Vec3) * 2);
		VoxelToWorld(voxelGridCoord[0], voxelGridCoord[1], voxelGridCoord[2], voxelWorldCoord, sceneBB, Nx, Ny, Nz);

		if (ray->dir.x < 0.0)
		{
			deltaT.x = -1 * Nx / ray->dir.x;
			tMax.x = (voxelWorldCoord[0].x - curpos.x) / ray->dir.x;
		}
		else if (ray->dir.x > 0.0)
		{
			deltaT.x = Nx / ray->dir.x;
			tMax.x = (voxelWorldCoord[1].x - curpos.x) / ray->dir.x;
		}

		if (ray->dir.y < 0.0)
		{
			deltaT.y = -1 * Ny / ray->dir.y;
			tMax.y = (voxelWorldCoord[0].y - curpos.y) / ray->dir.y;
		}
		else if (ray->dir.y > 0.0)
		{
			deltaT.y = Ny / ray->dir.y;
			tMax.y = (voxelWorldCoord[1].y - curpos.y) / ray->dir.y;
		}

		if (ray->dir.z < 0.0)
		{
			deltaT.z = -1 * Nz / ray->dir.z;
			tMax.z = (voxelWorldCoord[0].z - curpos.z) / ray->dir.z;
		}
		else if (ray->dir.z > 0.0)
		{
			deltaT.z = Nz / ray->dir.z;
			tMax.z = (voxelWorldCoord[1].z - curpos.z) / ray->dir.z;
		}

		if (ray->dir.x < 0.0) { stepX = -1; outX = -1; }
		if (ray->dir.y < 0.0) { stepY = -1; outY = -1; }
		if (ray->dir.z < 0.0) { stepZ = -1; outZ = -1; }

		while (1)
		{
			uniformGridIndex = voxelGridCoord[0] + voxelGridCoord[1] * xDim + voxelGridCoord[2] * xDim*yDim;

			if (uniformGrid[uniformGridIndex].size > 0)
			{
				for (int i = 0; i < uniformGrid[uniformGridIndex].size; i++)
				{
					if (IsRayTriangleIntersect(f[uniformGrid[uniformGridIndex].faceIndexSet[i]].v, *ray, f[uniformGrid[uniformGridIndex].faceIndexSet[i]].normalF, &rayObjectIntersectionPoint, &rayObjectDistance))
					{
						if (ray->distance > rayObjectDistance)
						{
							ray->isIntersect = true;
							ray->nearIntersectionPoint = rayObjectIntersectionPoint;
							ray->hitNormal = f[uniformGrid[uniformGridIndex].faceIndexSet[i]].normalF;
							ray->distance = rayObjectDistance;
							ray->objectColor = f[uniformGrid[uniformGridIndex].faceIndexSet[i]].color;
							ray->reflectivity = f[uniformGrid[uniformGridIndex].faceIndexSet[i]].reflectivity;
						}
					}
				}
			}

			if (ray->isIntersect)
			{
				isIntersect = true;
				break;
			}

			if (tMax.x < tMax.y && tMax.x < tMax.z)
			{
				voxelGridCoord[0] += stepX;
				if (voxelGridCoord[0] == outX)// out of voxel space
				{
					break;
				}
				tMax.x += deltaT.x;
			}
			else if (tMax.y < tMax.z)
			{
				voxelGridCoord[1] += stepY;
				if (voxelGridCoord[1] == outY)// out of voxel space
				{
					break;
				}
				tMax.y += deltaT.y;
			}
			else
			{
				voxelGridCoord[2] += stepZ;
				if (voxelGridCoord[2] == outZ)// out of voxel space
				{
					break;
				}
				tMax.z += deltaT.z;
			}
		}

		free(voxelGridCoord);
		free(voxelWorldCoord);
	}

	for (int i = 0; i < planeListSize; i++)
	{
		if (IsRayPlaneIntersect(planeList[i].A,
			planeList[i].B,
			planeList[i].C,
			planeList[i].D,
			*ray,
			&rayObjectIntersectionPoint,
			&rayObjectHitNormal,
			&rayObjectDistance))
		{
			isIntersect = true;
			if (ray->distance > rayObjectDistance)
			{
				ray->nearIntersectionPoint = rayObjectIntersectionPoint;
				ray->hitNormal = rayObjectHitNormal;
				ray->distance = rayObjectDistance;
				ray->reflectivity = planeList[i].reflectivity;

				ray->objectColor = planeList[i].color;
			}
		}
	}

	for (int i = 0; i < sphereListSize; i++)
	{
		if (IsRaySphereIntersect(sphereList[i].center,
			sphereList[i].radius,
			*ray,
			&rayObjectIntersectionPoint,
			&rayObjectHitNormal,
			&rayObjectDistance))
		{
			isIntersect = true;
			if (ray->distance > rayObjectDistance)
			{
				ray->nearIntersectionPoint = rayObjectIntersectionPoint;
				ray->hitNormal = rayObjectHitNormal;
				ray->distance = rayObjectDistance;
				ray->reflectivity = planeList[i].reflectivity;

				ray->objectColor = sphereList[i].color;
			}
		}
	}


	return isIntersect;
}


//Function Definitions
__global__ void RayTracer(Voxel* uniformGrid, Face* f, Plane* planeList, Sphere* sphereList, int planeListSize, int sphereListSize, Vec3* sensorGrid, Vec3* imageDataMat, SceneBB sceneBB, int x, int y, int xDim, int yDim, int zDim, int Nx, int Ny, int Nz)
{
	int blockOffset = threadIdx.x + threadIdx.y*blockDim.x;
	int gridOffset = blockIdx.x + blockIdx.y*gridDim.x;
	int index = blockOffset+ (gridOffset*blockDim.x*blockDim.y);

	if(index >= x*y)
		return;

	double Ka = 0.4, Kd = 0.8, Ks = 0.38, diffuseColor = 250, specularColor = 256, shininessVal = 90;
	
	Vec3 lightSource[1];
	Vec3 color;
		
	color.x = 0.0;
	color.y = 0.0;
	color.z = 0.0;

	lightSource[0] = Vector3(-995.0, 0.0, 0.0);
	//lightSource[1] = Vector3(405.0, 0.0, 0.0);

	Ray ray;
	
	ray.orig = Vector3(0, 0, -3.67);
	ray.dir = Normalize(Sub(ray.orig, sensorGrid[index]));
	ray.isIntersect = false;
	ray.distance = INFINITY;

	Vec3 lightVec, reflLightVec;
	double specular;
	float bias = 1.5;

	if(TraceRay(uniformGrid, f, planeList, sphereList, planeListSize, sphereListSize, &ray, sceneBB, x, y, xDim, yDim, zDim, Nx, Ny, Nz))
	{
		for (int i = 0; i < 2; i++)
		{
			//Shading Calculations
			lightVec = Sub(ray.nearIntersectionPoint, lightSource[i]);
			lightVec = Normalize(lightVec);

			float lambertian = DotProduct(lightVec, ray.hitNormal);

			if (lambertian < 0.0)
			{
				lambertian = 0.0;
			}

			color.x = (Kd * lambertian) * (ray.objectColor.x);
			color.y = (Kd * lambertian) * (ray.objectColor.y);
			color.z = (Kd * lambertian) * (ray.objectColor.z);
		}
		//Reflected Ray Tracing
		if (ray.reflectivity > 0)
		{
			Ray reflectedRay;
			reflectedRay.orig = Add(ray.nearIntersectionPoint, ScalarMul(ray.hitNormal, bias));
			reflectedRay.dir = reflect(ray.dir, ray.hitNormal);
			reflectedRay.isIntersect = false;
			reflectedRay.distance = INFINITY;

			if (!TraceRay(uniformGrid, f, planeList, sphereList, planeListSize, sphereListSize, &reflectedRay, sceneBB, x, y, xDim, yDim, zDim, Nx, Ny, Nz))
			{
				color.x += 10;
				color.y += 10;
				color.z += 10;
			}
			else
			{
				for (int i = 0; i < 2; i++)
				{
					lightVec = Sub(reflectedRay.nearIntersectionPoint, lightSource[i]);
					lightVec = Normalize(lightVec);

					float lambertian = DotProduct(lightVec, reflectedRay.hitNormal);

					if (lambertian < 0.0)
					{
						lambertian = 0.0;
					}

					color.x += (ray.reflectivity * 0.4 * (reflectedRay.objectColor.x * Kd * lambertian));
					color.y += (ray.reflectivity * 0.4 * (reflectedRay.objectColor.y * Kd * lambertian));
					color.z += (ray.reflectivity * 0.4 * (reflectedRay.objectColor.z * Kd * lambertian));
				}
			}
		}

		for (int i = 0; i < 2; i++)
		{
			//Shadow Ray Tracing
			Ray shadowRay;
			shadowRay.orig = Add(ray.nearIntersectionPoint, ScalarMul(ray.hitNormal, bias));
			shadowRay.dir = Normalize(Sub(shadowRay.orig, lightSource[i]));
			shadowRay.isIntersect = false;
			shadowRay.distance = INFINITY;

			float dist_lightSource = sqrt((shadowRay.orig.x - lightSource[i].x) * (shadowRay.orig.x - lightSource[i].x) +
				(shadowRay.orig.y - lightSource[i].y) * (shadowRay.orig.y - lightSource[i].y) +
				(shadowRay.orig.z - lightSource[i].z) * (shadowRay.orig.z - lightSource[i].z));

			if (TraceRay(uniformGrid, f, planeList, sphereList, planeListSize, sphereListSize, &shadowRay, sceneBB, x, y, xDim, yDim, zDim, Nx, Ny, Nz) && shadowRay.distance < dist_lightSource)
			{
				color.x = color.x / 2.0;//(1.0 + (shadowRay.distance / 1000 * shadowRay.distance / 1000));
				color.y = color.y / 2.0;//(1.0 + (shadowRay.distance / 1000 * shadowRay.distance / 1000));
				color.z = color.z / 2.0;//(1.0 + (shadowRay.distance / 1000 * shadowRay.distance / 1000));
			}
		}
	}

	imageDataMat[index] = color;
}