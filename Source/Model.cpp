#include "Model.h"

Model::Model(){}

Model::~Model(){}


__device__ __host__ float computeArea(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3)
{
	return 0.5*glm::length(glm::cross(v2 - v1, v3 - v1));
}

__device__ __host__ bool Model::computeIntersection(Ray* ray, int faceId)
{
	glm::vec3 vertex0 = verts[faces[faceId].v0];
	glm::vec3 vertex1 = verts[faces[faceId].v1];
	glm::vec3 vertex2 = verts[faces[faceId].v2];

	glm::vec3  edge1, edge2, h, s, q;
	float a, f, u, v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;
	
	h = glm::cross(ray->dir, edge2);
	a = glm::dot(edge1, h);
	if (a > -EPSILON && a < EPSILON) return false; 
	f = 1.0 / a;
	s = ray->orig - vertex0;
	u = f * glm::dot(s, h);
	if (u < 0.0 || u > 1.0) return false;
	q = glm::cross(s, edge1);
	v = f * glm::dot(ray->dir, q);
	if (v < 0.0 || u + v > 1.0) return false;
	
	float t = f * glm::dot(edge2, q);
	if (t > EPSILON)
	{
		ray->t = t;
		ray->hit_normal = glm::normalize(glm::cross(edge1, edge2));

		return true;
	}
	else return false;
}


void Model::createMesh(std::vector<glm::vec3> vertices, std::vector<glm::vec3> normals, std::vector<std::vector<int>> faces, glm::vec3 bbox_[2])
{
	this->n_verts = vertices.size();
	this->n_norms = normals.size();
	this->n_faces = faces.size();

	cudaError_t e1 = cudaMallocManaged(&verts, sizeof(glm::vec3)*n_verts);
	cudaDeviceSynchronize();
	cudaError_t e2 = cudaMallocManaged(&norms, sizeof(glm::vec3)*n_norms);
	cudaDeviceSynchronize();
	cudaError_t e3 = cudaMallocManaged(&(this->faces), sizeof(Triangle)*n_faces);
	cudaDeviceSynchronize();


	for (int i = 0; i < n_verts; i++)
	{
		verts[i] = vertices[i];
	}

	for (int i = 0; i < n_norms; i++)
	{
		norms[i] = normals[i];
	}

	this->bbox[0] = bbox_[0];
	this->bbox[1] = bbox_[1];

	for (int i = 0; i < n_faces; i++)
	{
		Triangle t;
		t.v0 = faces[i][0];
		t.v1 = faces[i][1];
		t.v2 = faces[i][2];

		t.bbox[0].x = glm::min(verts[t.v0].x, glm::min(verts[t.v1].x, verts[t.v2].x));
		t.bbox[0].y = glm::min(verts[t.v0].y, glm::min(verts[t.v1].y, verts[t.v2].y));
		t.bbox[0].z = glm::min(verts[t.v0].z, glm::min(verts[t.v1].z, verts[t.v2].z));

		t.bbox[1].x = glm::max(verts[t.v0].x, glm::max(verts[t.v1].x, verts[t.v2].x));
		t.bbox[1].y = glm::max(verts[t.v0].y, glm::max(verts[t.v1].y, verts[t.v2].y));
		t.bbox[1].z = glm::max(verts[t.v0].z, glm::max(verts[t.v1].z, verts[t.v2].z));

		this->faces[i] = t;
	}
}