#include "ObjectLoader.h"

#define FACE_SIZE 4096
typedef struct Voxel
{
	int size;
	int faceIndexSet[FACE_SIZE];
}Voxel;

Voxel* uniformGrid_host, *uniformGrid_device;

double Nx, Ny, Nz;
int xDim, yDim, zDim;

//Function Declarations
__host__ __device__ void VoxelToWorld(int x, int y, int z, Vec3* voxelVert, SceneBB sceneBB, int Nx, int Ny, int Nz);
__host__ __device__ void WorldToVoxel (Vec3 world, int* voxVert, SceneBB sceneBB, int Nx, int Ny, int Nz );
__global__ void InitializeUniformGrid(Voxel* uniformGrid, int uSize);
__global__ void GenerateUniformGrid(Face* f, int fSize, Voxel* uniformGrid, SceneBB sceneBB, int xDim, int yDim, int zDim, int Nx, int Ny, int Nz);


void GenerateUniformGridHost()
{
	for(int i = 0; i < f_host.size(); i++)
	{
		int s_cellX = floor((f_host[i].minBB.x - sceneBB.minP.x)/Nx);
		int e_cellX = floor((f_host[i].maxBB.x - sceneBB.minP.x)/Nx);

		int s_cellY = floor((f_host[i].minBB.y - sceneBB.minP.y)/Ny);
		int e_cellY = floor((f_host[i].maxBB.y - sceneBB.minP.y)/Ny);

		int s_cellZ = floor((f_host[i].minBB.z - sceneBB.minP.z)/Nz);
		int e_cellZ = floor((f_host[i].maxBB.z - sceneBB.minP.z)/Nz);

		int index;
		Vec3 voxVert[2];

		for(int x = s_cellX; x <= e_cellX; x++)
		{
			for(int y = s_cellY; y <= e_cellY; y++)
			{
				for(int z = s_cellZ; z <= e_cellZ; z++)
				{
					VoxelToWorld(x, y, z, voxVert, sceneBB, Nx, Ny, Nz);

					index = uniformGrid_host[x + y*xDim + z*xDim*yDim].size;
					uniformGrid_host[x + y*xDim + z*xDim*yDim].faceIndexSet[index] = i;
					uniformGrid_host[x + y*xDim + z*xDim*yDim].size++;

				}
			}
		}
	}
}

//Function Definitions
__host__ __device__ void WorldToVoxel (Vec3 world, int* voxVert, SceneBB sceneBB, int Nx, int Ny, int Nz)
{
	voxVert[0] = (int)( world.x - sceneBB.minP.x) / Nx; // component−wise division
	voxVert[1] = (int)( world.y - sceneBB.minP.y) / Ny; // component−wise division
	voxVert[2] = (int)( world.z - sceneBB.minP.z) / Nz; // component−wise division
}

__host__ __device__ void VoxelToWorld(int x, int y, int z, Vec3* voxelVert, SceneBB sceneBB, int Nx, int Ny, int Nz)
{
	voxelVert[0].x = sceneBB.minP.x + x*Nx;
	voxelVert[0].y = sceneBB.minP.y + y*Ny;
	voxelVert[0].z = sceneBB.minP.z + z*Nz;

	voxelVert[1].x = voxelVert[0].x + Nx;
	voxelVert[1].y = voxelVert[0].y + Ny;
	voxelVert[1].z = voxelVert[0].z + Nz;
}

__global__ void InitializeUniformGrid(Voxel* uniformGrid, int uSize)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if(index > uSize)
		return;

	uniformGrid[index].size = 0;
}

__global__ void GenerateUniformGrid(Face* f, int fSize, Voxel* uniformGrid, SceneBB sceneBB, int xDim, int yDim, int zDim, int Nx, int Ny, int Nz)
{
	int faceIndex = blockIdx.x*blockDim.x + threadIdx.x;

	if(faceIndex > fSize)
		return;

	int s_cellX = floor((f[faceIndex].minBB.x - sceneBB.minP.x)/Nx);
	int e_cellX = floor((f[faceIndex].maxBB.x - sceneBB.minP.x)/Nx);

	int s_cellY = floor((f[faceIndex].minBB.y - sceneBB.minP.y)/Ny);
	int e_cellY = floor((f[faceIndex].maxBB.y - sceneBB.minP.y)/Ny);

	int s_cellZ = floor((f[faceIndex].minBB.z - sceneBB.minP.z)/Nz);
	int e_cellZ = floor((f[faceIndex].maxBB.z - sceneBB.minP.z)/Nz);

	for(int x = s_cellX; x <= e_cellX; x++)
	{
		for(int y = s_cellY; y <= e_cellY; y++)
		{
			for(int z = s_cellZ; z <= e_cellZ; z++)
			{
				int index = uniformGrid[x + y*xDim + z*xDim*yDim].size;
				uniformGrid[x + y*xDim + z*xDim*yDim].faceIndexSet[index] = faceIndex;
				uniformGrid[x + y*xDim + z*xDim*yDim].size++;
			}
		}
	}
}
