#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#define GLM_FORCE_CUDA
#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/random.h>

#define INF 99999999.0
#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f
#define EPSILON2           0.001f

enum MaterialType
{
	emmissive,
	reflective,
	diffuse,
	refractive
};

typedef struct Ray
{
	glm::vec3 orig;
	glm::vec3 dir;
	glm::vec3 hit_normal;
	glm::vec3 color;
	glm::vec3 acc_color;
	MaterialType mat;
	double t;
	int remainingBounces;
	int pixel_index;
	thrust::default_random_engine rng;
}Ray;

typedef struct Triangle
{
	int v0, v1, v2;
	glm::vec3 bbox[2];
}Triangle;


class Model
{
public:
	glm::vec3* verts;
	glm::vec3* norms;
	glm::vec2* tex;
	Triangle* faces;
	float refractive_index;
	MaterialType mat;
	glm::vec3 color;

	int n_verts;
	int n_norms;
	int n_tex;
	int n_faces;
	

	glm::vec3 bbox[2];

	Model();
	~Model();
	void createMesh(std::vector<glm::vec3> vertices, std::vector<glm::vec3> normals, std::vector<std::vector<int>> faces, glm::vec3 bbox_[2]);
	__device__ __host__ bool computeIntersection(Ray* ray, int faceId);
};
