#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#define GLM_FORCE_CUDA
#include <vector>
#include <iostream>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/glew.h>
#include "UniformGrid.h"
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>

#define RIGHT 1
#define LEFT 2
#define UP 3
#define DOWN 4
#define IMG_X 400
#define IMG_Y 400

class Camera
{
public:
	glm::vec3 eye;
	glm::vec3 lookAt;
	glm::vec3 up;
	glm::vec3 forward;
	glm::vec3 right;
	glm::mat4 modelView;
	float moveDist;
	float field_of_view;
	float focal_length;

	glm::vec3* image;
	std::vector<UniformGrid> scene;
	UniformGrid* scene_gpu;
	Camera* cam_gpu;
	Ray* rays;
	int* dev_Stencil;

	Camera();
	~Camera();
	Camera(glm::vec3 eye, glm::vec3 lookAt, glm::vec3 up);
	void setCamera(glm::vec3 eye, glm::vec3 lookAt, glm::vec3 up);
	void rotateYaw(int dir);
	void rotatePitch(int dir);
	void moveForward();
	void moveBackward();
	void updateCam();
	__device__ __host__ glm::vec3 computePixelDir(float x, float y, int iter, int idx);
	void tracePath();
	void preProcess();
};

