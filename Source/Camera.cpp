#include "Camera.h"



Camera::Camera()
{
}

Camera::~Camera()
{
}

Camera::Camera(glm::vec3 eye_, glm::vec3 lookAt_, glm::vec3 up_)
{
	eye = glm::vec3(0.0, 10.0, -100.0);
	lookAt = glm::vec3(0, 0, 0);
	up = glm::vec3(0, 1, 0);
	forward = glm::normalize(lookAt - eye); 
	right = glm::normalize(glm::cross(forward, up));

	moveDist = 10.0f;
	field_of_view = (lookAt - eye).length(); 
	focal_length = 4.0;

	cudaError_t e = cudaMallocManaged(&image, sizeof(glm::vec3)*IMG_Y*IMG_X);
	cudaDeviceSynchronize();

	updateCam();
}

void Camera::setCamera(glm::vec3 eye_, glm::vec3 lookAt_, glm::vec3 up_)
{
	//this->eye = eye_;
	//this->lookAt = lookAt_;
	//this->up = up_;

	//field_of_view = (lookAt - eye).length();
	//updateCam();
}

void Camera::updateCam()
{
	glm::vec4 col1 = glm::vec4(right, 0.0);
	glm::vec4 col2 = glm::vec4(up, 0.0);
	glm::vec4 col3 = glm::vec4(forward, 0.0);
	glm::vec4 col4 = glm::vec4(eye, 1.0);

	modelView = glm::mat4(col1, col2, col3, col4);
}

void Camera::rotateYaw(int dir)
{
	if (cam_gpu == NULL) return;
	glm::vec3 moveDir;

	if (dir == RIGHT) moveDir = cam_gpu->right;
	else if (dir == LEFT) moveDir = glm::normalize(glm::cross(cam_gpu->up, -cam_gpu->forward));

	cam_gpu->lookAt = cam_gpu->lookAt + moveDir * moveDist;
	cam_gpu->forward = glm::normalize(cam_gpu->lookAt - cam_gpu->eye);

	float eyeDist = (cam_gpu->lookAt - cam_gpu->eye).length();
	if ((eyeDist - field_of_view) != 0.0f)
	{
		cam_gpu->lookAt = cam_gpu->lookAt - cam_gpu->forward * (eyeDist - field_of_view);
	}

	cam_gpu->right = glm::normalize(glm::cross(cam_gpu->up, cam_gpu->forward));
	//updateCam();
}

void Camera::rotatePitch(int dir)
{
	if (cam_gpu == NULL) return;
	glm::vec3 moveDir;

	if (dir == UP) moveDir = cam_gpu->up;
	else if (dir == DOWN) moveDir = glm::normalize(glm::cross(cam_gpu->forward, cam_gpu->right));


	cam_gpu->lookAt = cam_gpu->lookAt + moveDir * moveDist;
	cam_gpu->forward = glm::normalize(cam_gpu->lookAt - cam_gpu->eye);

	float eyeDist = (cam_gpu->lookAt - cam_gpu->eye).length();
	if ((eyeDist - field_of_view) != 0.0f)
	{
		cam_gpu->lookAt = cam_gpu->lookAt - cam_gpu->forward * (eyeDist - field_of_view);
	}

	cam_gpu->up = glm::normalize(glm::cross(cam_gpu->right, cam_gpu->forward));
	//updateCam();
}

void Camera::moveForward()
{
	cam_gpu->eye = cam_gpu->eye + cam_gpu->forward * 5.0f;
	field_of_view = (cam_gpu->lookAt - cam_gpu->eye).length();
	//updateCam();
}

void Camera::moveBackward()
{
	cam_gpu->eye = cam_gpu->eye - cam_gpu->forward * 5.0f;
	field_of_view = (cam_gpu->lookAt - cam_gpu->eye).length();
	//updateCam();
}


__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

__device__ glm::vec3 random_in_unit_disk(thrust::default_random_engine& rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec3 p = glm::vec3(u01(rng), u01(rng), 0);
	p = glm::normalize(p);

	return p;
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
	int h = utilhash((1 << 31) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__device__ __host__ glm::vec3 Camera::computePixelDir(float x, float y, int iter, int idx)
{
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
	thrust::uniform_real_distribution<float> u01(-0.5, 0.5);

	float x_offset = u01(rng), y_offset = u01(rng);
	float tx = (x - 0.5*IMG_X + x_offset)*0.0099;
	float ty = (y - 0.5*IMG_Y + y_offset)*0.0099;

	glm::vec3 imagePt = eye + forward * focal_length + right * tx + up * ty;

	return glm::normalize(imagePt - eye);
}


__device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine& rng) 
{	
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); 
	float over = sqrt(1 - up * up); 
	float around = u01(rng) * TWO_PI;

	glm::vec3 directionNotNormal;

	if (abs(normal.x) < SQRT_OF_ONE_THIRD) 
	{
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) 
	{
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else 
	{
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal + cos(around) * over * perpendicularDirection1 + sin(around) * over * perpendicularDirection2;
}

__global__ void computeIntersections(UniformGrid* scene, int n_scene, Ray* rays, int n_rays)
{
	int ri = blockIdx.x * blockDim.x + threadIdx.x;

	double tmin = INF;
	glm::vec3 intersectionPt;
	glm::vec3 hitNormal;
	glm::vec3 color;
	MaterialType mat;

	rays[ri].t = INF;
	color = glm::vec3(0.0f);

	for (int i = 0; i < n_scene; i++)
	{
		if (scene[i].computeIntersection(&rays[ri]))
		{
			if (rays[ri].t < tmin)
			{
				tmin = rays[ri].t;
				hitNormal = rays[ri].hit_normal;
				color = scene[i].model->color;
				mat = scene[i].model->mat;
			}
		}
	}

	if (tmin < INF)
	{
		rays[ri].t = tmin;
		rays[ri].hit_normal = hitNormal;
		rays[ri].color = color;
		rays[ri].mat = mat;
		return;
	}
	//rays[ri].t = -1.0;
}

__device__ void scatterRay(Ray& ray, thrust::default_random_engine& rng) 
{
	glm::vec3 intersectionPt = ray.orig + ray.dir*(float)(ray.t - EPSILON);

	if (ray.mat == reflective)
	{
		ray.acc_color *= ray.color; 

		ray.dir = glm::normalize(glm::reflect(ray.dir, ray.hit_normal));
		ray.orig = intersectionPt + ray.dir * EPSILON;
		
	}
	else if (ray.mat == diffuse)
	{
		ray.acc_color *= ray.color;

		glm::vec3 scatter_direction;
		scatter_direction = glm::normalize(calculateRandomDirectionInHemisphere(glm::normalize(ray.hit_normal), rng));

		ray.dir = scatter_direction;
		ray.orig = intersectionPt + ray.dir * EPSILON;
		
	}

	ray.remainingBounces--;
}

__global__ void shadeBSDFMaterial(int iter, int nRays, int depth, Ray* rays)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nRays && rays[idx].remainingBounces > 0)
	{
		if (rays[idx].t > 0.0f)
		{ 
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			if (rays[idx].mat == emmissive)
			{
				rays[idx].acc_color *= rays[idx].color*5.0f;
				rays[idx].remainingBounces = 0;
			}
			else 
			{
				scatterRay(rays[idx], rng);
			}
		}
		else 
		{
			rays[idx].acc_color = glm::vec3(0.0f);
			rays[idx].remainingBounces = 0;
		}
	}
}


void Camera::preProcess()
{
	cudaMallocManaged(&cam_gpu, sizeof(Camera));
	cudaError_t e = cudaDeviceSynchronize();

	cam_gpu->eye = this->eye;
	cam_gpu->field_of_view = this->field_of_view;
	cam_gpu->forward = this->forward;
	cam_gpu->up = this->up;
	cam_gpu->right = this->right;
	cam_gpu->lookAt = this->lookAt;
	cam_gpu->modelView = this->modelView;
	cam_gpu->focal_length = this->focal_length;

	cudaMallocManaged(&scene_gpu, sizeof(UniformGrid)*scene.size());
	cudaDeviceSynchronize();

	for (int i = 0; i < scene.size(); i++)
	{
		scene_gpu[i] = scene[i];
	}

	int nRays = IMG_X*IMG_Y;
	cudaMallocManaged(&rays, sizeof(Ray)*nRays);
	cudaDeviceSynchronize();

	cudaMallocManaged(&dev_Stencil, nRays * sizeof(int));

	for (int y = 0; y < IMG_Y; y++)
	{
		for (int x = 0; x < IMG_X; x++)
		{
			int ind = x + y * IMG_X;
			image[ind] = glm::vec3(0);
		}
	}


}

struct hasTerminated
{
	__host__ __device__
		bool operator()(const int& x)
	{
		return x == 1;
	}
};

__global__ void CompactionStencil(int nRays, Ray* rays, int* dev_Stencil)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nRays)
	{
		if (rays[index].remainingBounces == 0)
		{
			dev_Stencil[index] = 0;
			return;
		}
		dev_Stencil[index] = 1;
	}
}

__global__ void generateImage(int nPixels, glm::vec3* image, Ray* rays)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPixels)
	{
		Ray ray = rays[index];
		image[ray.pixel_index] += ray.acc_color*0.01f;
	}
}

__global__ void saveImage(glm::vec3* image, int samples, int nPixels) {
	

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPixels)
	{
		glm::vec3 pix = image[index];

		image[index].x = glm::clamp((int)(pix.x / samples * 255), 0, 255);
		image[index].y = glm::clamp((int)(pix.y / samples * 255), 0, 255);
		image[index].z = glm::clamp((int)(pix.z / samples * 255), 0, 255);
	}
}

__global__ void generateRays(Ray* rays, Camera* cam_gpu, int iter)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < IMG_X && y < IMG_Y) {
		int index = x + (y * IMG_X);

		Ray ray;
		ray.orig = cam_gpu->eye;
		ray.dir = cam_gpu->computePixelDir(x, y, iter, index);
		ray.t = INF;
		ray.color = glm::vec3(1);
		ray.acc_color = glm::vec3(1);
		ray.pixel_index = index;
		ray.remainingBounces = 8;
		rays[index] = ray;
	}
}

void Camera::tracePath()
{
	const int blockSize1d = 128;
	int depth = 0;
	int nRays = IMG_X*IMG_Y;
	int n_scene = scene.size();

	int SAMPLES = 100;

	for (int iter = 0; iter < SAMPLES; iter++)
	{
		bool iterationComplete = false;
		nRays = IMG_X*IMG_Y;

		const dim3 blockSize2d(8, 8);
		const dim3 blocksPerGrid2d((IMG_X + blockSize2d.x - 1) / blockSize2d.x,(IMG_Y + blockSize2d.y - 1) / blockSize2d.y);
		generateRays << <blocksPerGrid2d, blockSize2d >> >(rays, cam_gpu, iter);
		cudaDeviceSynchronize();

		while (!iterationComplete)
		{
			// tracing
			dim3 numblocksPathSegmentTracing = (nRays + blockSize1d - 1) / blockSize1d;
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (scene_gpu, n_scene, rays, nRays);
			cudaDeviceSynchronize();

			shadeBSDFMaterial << < numblocksPathSegmentTracing, blockSize1d >> > (iter, nRays, depth, rays);
			cudaDeviceSynchronize();

			CompactionStencil << < numblocksPathSegmentTracing, blockSize1d >> > (nRays, rays, dev_Stencil);
			cudaDeviceSynchronize();

			Ray* itr = thrust::stable_partition(thrust::device, rays, rays + nRays, dev_Stencil, hasTerminated());
			int n = itr - rays;
			nRays = n;

			if (nRays == 0)
			{
				iterationComplete = true;
			}
		}

		dim3 numBlocksPixels = (IMG_X*IMG_Y + blockSize1d - 1) / blockSize1d;
		generateImage << <numBlocksPixels, blockSize1d >> > (IMG_X*IMG_Y, image, rays);
		cudaDeviceSynchronize();
	}
	/*dim3 numBlocksPixels = (IMG_X*IMG_Y + blockSize1d - 1) / blockSize1d;
	saveImage << <numBlocksPixels, blockSize1d >> >(image, SAMPLES, IMG_X*IMG_Y);
	cudaDeviceSynchronize();*/
}