#include "DisplayMenu.h"


//Function Declarations
void AllocateMemoryOnGPU();
void CudaCleanUp();
Vec3* GenerateSensorGrid(int x_pixels, int y_pixels, Vec3 center);
void VisualizeResults(Vec3* imageDataMat, int x, int y);

int main()
{
	focalPoint = Vector3(0, 0, -FOCAL_LENGTH);
	
	sensorGrid_host = GenerateSensorGrid(XMAX, YMAX, focalPoint);
	imageData_host = (Vec3*)malloc(XMAX*YMAX*sizeof(Vec3));

	DisplayMenu();
	AllocateMemoryOnGPU();

	printf("Ray Tracing Started\n");

	clock_t start = clock();

	dim3 block_3(32,4,1);
	dim3 grid_3((int)ceil((float)XMAX/block_3.x), (int)ceil((float)YMAX/block_3.y), 1);
	
	RayTracer<<<grid_3, block_3>>>(	uniformGrid_device, 
									f_device,
									planeList,
									sphereList,
									planeListSize,
									sphereListSize,
									sensorGrid_device, 
									imageData_device, 
									sceneBB, 
									XMAX, 
									YMAX, 
									xDim, 
									yDim, 
									zDim, 
									Nx, 
									Ny, 
									Nz);
	cudaDeviceSynchronize();

	clock_t stop = clock();

	double timeInsecs = (stop-start)*1000/CLOCKS_PER_SEC;
	printf("Ray Tracing Done: Time taken: %lf\n", timeInsecs/1000);

	cudaError_t err = cudaMemcpy(imageData_host, imageData_device, XMAX*YMAX*sizeof(Vec3), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		printf("cudaMemcpy() for imageData_device: %s", cudaGetErrorString(err));
	}

	VisualizeResults(imageData_host, XMAX, YMAX);

	CudaCleanUp();
}

void AllocateMemoryOnGPU()
{
	cudaError_t err;
	uSize = xDim*yDim*zDim;
	fSize = f_host.size();

	Face* f_host_arr = (Face*)malloc(fSize*sizeof(Face));

	for(int i = 0; i < f_host.size(); i++)
	{
		f_host_arr[i].v[0] = f_host[i].v[0];
		f_host_arr[i].v[1] = f_host[i].v[1];
		f_host_arr[i].v[2] = f_host[i].v[2];

		f_host_arr[i].normalF = f_host[i].normalF;
		f_host_arr[i].reflectivity = f_host[i].reflectivity;
		f_host_arr[i].color = f_host[i].color;
		f_host_arr[i].minBB  = f_host[i].minBB;
		f_host_arr[i].maxBB = f_host[i].maxBB;
	}

	Plane* planeList_host = (Plane*)malloc(sizeof(Plane)*planeListHolder.size());
	Sphere* sphereList_host = (Sphere*)malloc(sizeof(Sphere)*sphereListHolder.size());

	for (int i = 0; i < planeListHolder.size(); i++)
	{
		planeList_host[i] = planeListHolder[i];
	}

	for (int i = 0; i < sphereListHolder.size(); i++)
	{
		sphereList_host[i] = sphereListHolder[i];
	}

	planeListSize = planeListHolder.size();
	sphereListSize = sphereListHolder.size();

	err = cudaMalloc((void**)&planeList, planeListHolder.size() * sizeof(Plane));
	if (err != cudaSuccess)
	{
		printf("cudaMalloc() for planeList: %s", cudaGetErrorString(err));
	}
	err = cudaMemcpy(planeList, planeList_host, planeListHolder.size() * sizeof(Plane), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("cudaMemcpy() for planeList: %s", cudaGetErrorString(err));
	}

	err = cudaMalloc((void**)&sphereList, sphereListHolder.size() * sizeof(Sphere));
	if (err != cudaSuccess)
	{
		printf("cudaMalloc() for sphereList: %s", cudaGetErrorString(err));
	}
	err = cudaMemcpy(sphereList, sphereList_host, sphereListHolder.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("cudaMemcpy() for sphereList: %s", cudaGetErrorString(err));
	}

	err = cudaMalloc((void**)&sensorGrid_device, XMAX*YMAX*sizeof(Vec3));
	if(err != cudaSuccess)
	{
		printf("cudaMalloc() for sensorGrid_device: %s",cudaGetErrorString(err));
	}
	err = cudaMemcpy(sensorGrid_device, sensorGrid_host, XMAX*YMAX*sizeof(Vec3), cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("cudaMemcpy() for sensorGrid_device: %s",cudaGetErrorString(err));
	}

	err = cudaMalloc((void**)&imageData_device, XMAX*YMAX*sizeof(Vec3));
	if(err != cudaSuccess)
	{
		printf("cudaMalloc() for imageData_device: %s",cudaGetErrorString(err));
	}
	
	err = cudaMalloc((void**)&f_device, fSize*sizeof(Face));
	if(err != cudaSuccess)
	{
		printf("cudaMalloc() for f_device: %s",cudaGetErrorString(err));
	}
	err = cudaMemcpy(f_device, f_host_arr, fSize*sizeof(Face), cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("cudaMemcpy() for f_device: %s",cudaGetErrorString(err));
	}

	err = cudaMalloc((void**)&uniformGrid_device, uSize*sizeof(Voxel));
	if(err != cudaSuccess)
	{
		printf("cudaMalloc() for uniformGrid_device: %s",cudaGetErrorString(err));
	}
	err = cudaMemcpy(uniformGrid_device, uniformGrid_host, uSize*sizeof(Voxel), cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("cudaMemcpy() for uniformGrid_device: %s",cudaGetErrorString(err));
	}
	err = cudaMemcpy(uniformGrid_device, uniformGrid_host, uSize*sizeof(Voxel), cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("cudaMemcpy() for uniformGrid_device: %s",cudaGetErrorString(err));
	}

	free(f_host_arr);
	f_host.clear();
}

void CudaCleanUp()
{
	if(uniformGrid_device != NULL)
		cudaFree(uniformGrid_device);

	if(f_device != NULL)
		cudaFree(f_device);

	if(imageData_device != NULL)
		cudaFree(imageData_device);

	if(sensorGrid_device != NULL)
		cudaFree(sensorGrid_device);

	if (planeList != NULL)
		cudaFree(planeList);

	if (sphereList != NULL)
		cudaFree(sphereList);
}

Vec3* GenerateSensorGrid(int x_pixels, int y_pixels, Vec3 center)
{
	Vec3* matrix = (Vec3*)malloc(sizeof(Vec3)*y_pixels*x_pixels);
	int matCenterX = x_pixels / 2, matCenterY = y_pixels / 2; //Corresponding matrix indices for the "center" coordinate

	for (int i = 0; i < y_pixels; i++)
	{
		for (int j = 0; j < x_pixels; j++)
		{
			matrix[i*x_pixels + j].x = center.x + (j - matCenterX)*0.05;
			matrix[i*x_pixels + j].y = center.y + (matCenterY - i)*0.05;
			matrix[i*x_pixels + j].z = 0;
		}
	}

	return matrix;
}

void VisualizeResults(Vec3* imageDataMat, int x, int y)
{
	Mat img(y, x, CV_8UC3, Scalar(250,250,250));

	if(img.empty())
	{
		printf("Image Not Created");
	}
	else
	{
		for(int i = 0; i < y; i++)
		{
			for(int j = 0; j < x; j++)
			{
				img.at<Vec3b>(i, j)[0] = imageDataMat[i*x + j].z;
				img.at<Vec3b>(i, j)[1] = imageDataMat[i*x + j].y;
				img.at<Vec3b>(i, j)[2] = imageDataMat[i*x + j].x;
			
			}
		}
	}

	char imageName[50] = "IMAGE";

	namedWindow(imageName, WINDOW_AUTOSIZE );
	imshow(imageName, img );
	waitKey(0);
	imwrite( "Image.jpg", img );
}

