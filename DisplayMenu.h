#include "RayTracer.h"

void DisplayMenu()
{
	int choice = 1;

	while (choice)
	{
		printf("_________________\n");
		printf("| Plane     | 1 |\n");
		printf("|           |   |\n");
		printf("| Sphere    | 2 |\n");
		printf("|           |   |\n");
		printf("| 3D Model  | 3 |\n");
		printf("|           |   |\n");
		printf("| Exit      | 0 |\n");
		printf("|___________|___|\n");

		scanf("%d", &choice);

		switch (choice)
		{
			case 1:
			{
				Plane plane;

				printf("A: ");
				scanf("%f", &plane.A);
				printf("B: ");
				scanf("%f", &plane.B);
				printf("C: ");
				scanf("%f", &plane.C);
				printf("D: ");
				scanf("%f", &plane.D);

				printf("Reflectivity:");
				scanf("%f", &plane.reflectivity);

				printf("Color:\n");
				printf("Red Component: ");
				scanf("%f", &plane.color.x);
				printf("Green Component: ");
				scanf("%f", &plane.color.y);
				printf("Blue Component: ");
				scanf("%f", &plane.color.z);

				planeListHolder.push_back(plane);
				break;
			}
			case 2:
			{
				Sphere sphere;

				printf("Radius: ");
				scanf("%f", &sphere.radius);
				printf("CenterX: ");
				scanf("%f", &sphere.center.x);
				printf("CenterY: ");
				scanf("%f", &sphere.center.y);
				printf("CenterZ: ");
				scanf("%f", &sphere.center.z);

				printf("Reflectivity:");
				scanf("%f", &sphere.reflectivity);

				printf("Color:\n");
				printf("Red Component: ");
				scanf("%f", &sphere.color.x);
				printf("Green Component: ");
				scanf("%f", &sphere.color.y);
				printf("Blue Component: ");
				scanf("%f", &sphere.color.z);

				sphereListHolder.push_back(sphere);
				break;
			}
			case 3:
			{
				ReadOBJFile();

				double dx = abs(sceneBB.maxP.x - sceneBB.minP.x);
				double dy = abs(sceneBB.maxP.y - sceneBB.minP.y);
				double dz = abs(sceneBB.maxP.z - sceneBB.minP.z);

				printf("Bounding Box Dimensions %lf, %lf %lf\n", sceneBB.minP.x, sceneBB.minP.y, sceneBB.minP.z);
				printf("Bounding Box Dimensions %lf, %lf %lf\n", sceneBB.maxP.x, sceneBB.maxP.y, sceneBB.maxP.z);

				/*
				double V =  dx * dy * dz;
				int P = f_host.size();
				int k = 5;

				double K = k*P/V;
				double cbrt = pow(K, 1/3.);

				Nx = ceil(dx*cbrt);
				Ny = ceil(dy*cbrt);
				Nz = ceil(dz*cbrt);
				*/

				Nx = 50.0;
				Ny = 50.0;
				Nz = 50.0;

				xDim = (int)(ceil(dx / Nx) + 1);
				yDim = (int)(ceil(dy / Ny) + 1);
				zDim = (int)(ceil(dz / Nz) + 1);

				printf("X: %d, Y: %d Z: %d\n", xDim, yDim, zDim);

				uniformGrid_host = (Voxel*)malloc(xDim*yDim*zDim * sizeof(Voxel));

				for (int i = 0; i < xDim*yDim*zDim; i++)
				{
					uniformGrid_host[i].size = 0;
				}
				printf("UniformGrid Initialized\n");

				GenerateUniformGridHost();
				printf("UniformGrid Populated\n");
				break;
			}

		}
	}
}


