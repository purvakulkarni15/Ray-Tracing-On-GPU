#pragma once
#include <vector>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Model.h"
#include "ModelLoader.h"
#include <limits>


typedef struct Voxel
{
	int triangleIds[100];
	int size;
	__device__ __host__ bool computeIntersection(Ray* ray, Model* model);
}Voxel;

class UniformGrid
{
public:

	Model* model;
	Voxel* grid;
	double voxel_sz_x, voxel_sz_y, voxel_sz_z;
	int GRID_X, GRID_Y, GRID_Z;

	UniformGrid();
	UniformGrid(ModelLoader modelLoader, int i, int g_sz);
	~UniformGrid();

	void addObject();
	__device__ __host__ bool computeIntersection(Ray* ray);
	__device__ __host__ bool getRayBoxIntersection(Ray* ray);
};

