#include "UniformGrid.h"

__device__ __host__ bool Voxel::computeIntersection(Ray* ray, Model* model)
{
	glm::vec3 hitNormal;
	float tmin = INF;

	for (int i = 0; i < size; i++)
	{
		int faceId = triangleIds[i];

		if (model->computeIntersection(ray, faceId))
		{
			if (ray->t < tmin)
			{
				tmin = ray->t;
				hitNormal = ray->hit_normal;
			}
		}
	}

	if (tmin < INF)
	{
		ray->t = tmin;
		ray->hit_normal = hitNormal;
		return true;
	}
	return false;
}

UniformGrid::UniformGrid() {}

UniformGrid::UniformGrid(ModelLoader modelLoader, int i, int g_sz)
{

	cudaMallocManaged(&model, sizeof(Model));
	cudaDeviceSynchronize();

	model->createMesh(modelLoader.meshes[i].vertices, modelLoader.meshes[i].normals, modelLoader.meshes[i].faces, modelLoader.meshes[i].aabb);

	voxel_sz_x = 10.0;
	voxel_sz_y = 10.0;
	voxel_sz_z = 10.0;

	GRID_X = glm::ceil((model->bbox[1].x - model->bbox[0].x) / voxel_sz_x);
	GRID_Y = glm::ceil((model->bbox[1].y - model->bbox[0].y) / voxel_sz_y);
	GRID_Z = glm::ceil((model->bbox[1].z - model->bbox[0].z) / voxel_sz_z);


	int grid_sz = GRID_X * GRID_Y * GRID_Z;
	cudaMallocManaged(&grid, sizeof(Voxel)*grid_sz);
	cudaDeviceSynchronize();
	//size_t free, total;
	//cudaMemGetInfo(&free, &total);

	for (int i = 0; i < grid_sz; i++)
	{
		grid[i].size = 0;
	}
}

UniformGrid::~UniformGrid(){}

void UniformGrid::addObject()
{
	for (int i = 0; i < model->n_faces; i++)
	{
		Triangle* tri = &model->faces[i];

		int x_min = glm::floor((tri->bbox[0].x - model->bbox[0].x) / voxel_sz_x);
		int y_min = glm::floor((tri->bbox[0].y - model->bbox[0].y) / voxel_sz_y);
		int z_min = glm::floor((tri->bbox[0].z - model->bbox[0].z) / voxel_sz_z);

		int x_max = glm::floor((tri->bbox[1].x - model->bbox[0].x) / voxel_sz_x);
		int y_max = glm::floor((tri->bbox[1].y - model->bbox[0].y) / voxel_sz_y);
		int z_max = glm::floor((tri->bbox[1].z - model->bbox[0].z) / voxel_sz_z);


		for (int x = x_min; x <= x_max; x++)
		{
			for (int y = y_min; y <= y_max; y++)
			{
				for (int z = z_min; z <= z_max; z++)
				{
					int index = x + y * GRID_X + z * GRID_X*GRID_Y;
					
					grid[index].triangleIds[grid[index].size] = i;
					grid[index].size++;
					
					if (grid[index].size > 100)
					{
						grid[index].size = 0;
					}
				}
			}
		}
	}

}

__device__ __host__ void swap(float& a, float& b)
{
	float t = a;
	a = b;
	b = t;
}

__device__ __host__ bool UniformGrid::getRayBoxIntersection(Ray* ray)
{
	float tmin = 0, tmax = 0;
	
	//if (ray->dir.x != 0)
	{
	tmin = (model->bbox[0].x - ray->orig.x) / ray->dir.x;
	tmax = (model->bbox[1].x - ray->orig.x) / ray->dir.x;
	}

	if (tmin > tmax) swap(tmin, tmax);

	float tymin = 0, tymax = 0;

	//if (ray->dir.y != 0)
	{
		tymin = (model->bbox[0].y - ray->orig.y) / ray->dir.y;
		tymax = (model->bbox[1].y - ray->orig.y) / ray->dir.y;
	}

	if (tymin > tymax) swap(tymin, tymax);

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = 0, tzmax = 0;

	//if (ray->dir.z != 0)
	{

		tzmin = (model->bbox[0].z - ray->orig.z) / ray->dir.z;
		tzmax = (model->bbox[1].z - ray->orig.z) / ray->dir.z;
	}

	if (tzmin > tzmax) swap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin > tmax) swap(tmin, tmax);

	if (tmin > 0.0f) ray->t = tmin;

	return true;
}

__device__ __host__ bool UniformGrid::computeIntersection(Ray* ray)
{
	glm::vec3 currPos;
	//if (!getRayBoxIntersection(ray)) return false;
	//else
	{
		glm::vec3 hitNormal;
		float tmin = INF;

		for (int i = 0; i < model->n_faces; i++)
		{
			if (model->computeIntersection(ray, i))
			{
				if (ray->t < tmin)
				{
					tmin = ray->t;
					hitNormal = ray->hit_normal;
				}
			}
		}

		if (tmin < INF)
		{
			ray->t = tmin;
			ray->hit_normal = hitNormal;
			return true;
		}
		return false;
	}
	
	currPos = ray->orig + ray->dir*(float)ray->t;

	currPos += glm::vec3(0.001, 0.001, 0.001);

	int curr[3];

	if ((currPos.x - model->bbox[0].x) < 0)
	{
		return false;
	}
	if ((currPos.y - model->bbox[0].y) < 0)
	{
		return false;
	}
	if ((currPos.z - model->bbox[0].z) < 0)
	{
		return false;
	}

	curr[0] = glm::floor((currPos.x - model->bbox[0].x) / voxel_sz_x);
	curr[1] = glm::floor((currPos.y - model->bbox[0].y) / voxel_sz_y);
	curr[2] = glm::floor((currPos.z - model->bbox[0].z) / voxel_sz_z);
	
	glm::vec3 tMax = glm::vec3(99999.0f, 99999.0f, 99999.0f);
	glm::vec3 delta = glm::vec3(99999.0f, 99999.0f, 99999.0f);

	int step_x = 1, step_y = 1, step_z = 1;
	int out_x = GRID_X, out_y = GRID_Y, out_z = GRID_Z;

	float nextPosX = model->bbox[0].x + (curr[0] + 1)*voxel_sz_x;
	float prevPosX = model->bbox[0].x + (curr[0])*voxel_sz_x;
	
	if (ray->dir.x > 0.0f)
	{
		delta.x = voxel_sz_x / ray->dir.x;
		tMax.x = (nextPosX - currPos.x) / ray->dir.x;
		step_x = 1;
		out_x = GRID_X;
	}
	else if (ray->dir.x < 0.0f)
	{
		delta.x = -1.0f*voxel_sz_x / ray->dir.x;
		tMax.x = (prevPosX - currPos.x) / ray->dir.x;
		step_x = -1;
		out_x = -1;
	}

	float nextPosY = model->bbox[0].y + (curr[1] + 1)*voxel_sz_y;
	float prevPosY = model->bbox[0].y + (curr[1])*voxel_sz_y;

	if (ray->dir.y > 0.0f)
	{
		delta.y = voxel_sz_y / ray->dir.y;
		tMax.y = (nextPosY - currPos.y) / ray->dir.y;
		step_y = 1;
		out_y = GRID_Y;
	}
	else if (ray->dir.y < 0.0f)
	{
		delta.y = -1.0f*voxel_sz_y / ray->dir.y;
		tMax.y = (prevPosY - currPos.y) / ray->dir.y;
		step_y = -1;
		out_y = -1;
	}

	float nextPosZ = model->bbox[0].z + (curr[2] + 1)*voxel_sz_z;
	float prevPosZ = model->bbox[0].z + (curr[2])*voxel_sz_z;

	if (ray->dir.z > 0.0f)
	{
		delta.z = voxel_sz_z / ray->dir.z;
		tMax.z = (nextPosZ - currPos.z) / ray->dir.z;
		step_z = 1;
		out_z = GRID_Z;
	}
	else if (ray->dir.z < 0.0f)
	{
		delta.z = -1.0f*voxel_sz_z / ray->dir.z;
		tMax.z = (prevPosZ - currPos.z) / ray->dir.z;
		step_z = -1;
		out_z = -1;
	}

	while (1)
	{
		int index = curr[0] + curr[1] * GRID_X + curr[2] * GRID_X * GRID_Y;

		if (grid[index].computeIntersection(ray, model))
		{
			return true;
		}

		if (tMax.x < tMax.y && tMax.x < tMax.z)
		{
			curr[0] += step_x;
			if (curr[0] == out_x)
			{
				return false;
			}
			tMax.x += delta.x;
		}
		else if (tMax.y < tMax.z)
		{
			curr[1] += step_y;
			if (curr[1] == out_y)
			{
				return false;
			}
			tMax.y += delta.y;
		}
		else
		{
			curr[2] += step_z;
			if (curr[2] == out_z)
			{
				return false;
			}
			tMax.z += delta.z;
		}
	}
}
