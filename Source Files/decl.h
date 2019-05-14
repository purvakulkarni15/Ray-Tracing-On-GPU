#include"cuda_runtime.h"
#include<device_launch_parameters.h>
#include<cuda.h>
#include<stdio.h>
#include<conio.h>
#include<vector>
#include<time.h>
#include<math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"	

using namespace std;
using namespace cv;

#define XMAX 640
#define YMAX 480
#define FOCAL_LENGTH 3.67

typedef struct Vector3f
{
	float x; 
	float y;
	float z;
}Vec3;

typedef struct Face
{
	Vec3 v[3];
	//Vec3 vt[2];
	//Vec3 vn[3];
	Vec3 color;
	float reflectivity;
	Vec3 normalF;
	Vec3 minBB, maxBB;
}Face;

typedef struct SceneBB
{
	Vec3 minP, maxP;
}SceneBB;


typedef struct Ray
{
	Vec3 dir;
	Vec3 orig;
	Vec3 nearIntersectionPoint;
	Vec3 hitNormal;
	Vec3 objectColor;
	float distance;
	float reflectivity;
	bool isIntersect;
}Ray;

typedef struct Plane
{
	float A, B, C, D;
	float reflectivity;
	Vec3 color;
}Plane;

typedef struct Sphere
{
	float radius;
	Vec3 center;
	float reflectivity;
	Vec3 color;
}Sphere;

vector<Plane> planeListHolder;
vector<Sphere> sphereListHolder;

Plane* planeList;
Sphere* sphereList;
int planeListSize, sphereListSize;


vector<Vec3> v; //Model Vertices
vector<Vec3> vt;//Model Texture Coordinates
vector<Vec3> vn;//Model Vertex Normals
vector<Face> f_host;


//Global Variables
Face* f_device;
SceneBB sceneBB;
Vec3* sensorGrid_host, *sensorGrid_device;
Vec3* imageData_host, *imageData_device;
Vec3 focalPoint;
int uSize, fSize;

