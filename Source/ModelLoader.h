#pragma once
#include <vector>
#include <iostream>
#include <assimp/Importer.hpp>      
#include <assimp/scene.h> 
#include <assimp/postprocess.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace std;

struct MeshData {

	vector<glm::vec3> vertices;
	vector<glm::vec3> normals;
	vector<glm::vec2> textures;
	vector<vector<int>> faces;
	glm::vec3 aabb[2];


};

class ModelLoader
{
public:
	vector<MeshData> meshes;
	ModelLoader();
	~ModelLoader();

	void loadModel(string path);
	void processNode(aiNode *node, const aiScene *scene);
	MeshData processMesh(aiMesh *mesh, const aiScene *scene);
};

