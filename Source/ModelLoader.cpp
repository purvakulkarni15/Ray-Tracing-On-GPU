#include "ModelLoader.h"

ModelLoader::ModelLoader() {}

ModelLoader::~ModelLoader() {}

void ModelLoader::loadModel(string path)
{
	Assimp::Importer import;
	const aiScene *scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		cout << "ERROR::ASSIMP::" << import.GetErrorString() << endl;
		return;
	}

	processNode(scene->mRootNode, scene);
}

void ModelLoader::processNode(aiNode *node, const aiScene *scene)
{
	for (unsigned int i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(mesh, scene));
	}

	for (unsigned int i = 0; i < node->mNumChildren; i++)
	{
		processNode(node->mChildren[i], scene);
	}
}

MeshData ModelLoader::processMesh(aiMesh *mesh, const aiScene *scene)
{
	MeshData meshData;

	float scale = 10.0f;
	
	meshData.aabb[0] = glm::vec3(99999.0f, 99999.0f, 99999.0f);
	meshData.aabb[1] = glm::vec3(-99999.0f, -99999.0f, -99999.0f);

	for (int i = 0; i < mesh->mNumVertices; i++)
	{
		meshData.vertices.push_back(scale*glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
		meshData.normals.push_back(scale*glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
		if (mesh->mTextureCoords[0])
		{
			meshData.textures.push_back(scale*glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y));
		}
		else
		{
			meshData.textures.push_back(glm::vec2(0.0f, 0.0f));
		}

		meshData.aabb[0].x = std::min(meshData.aabb[0].x, meshData.vertices[i].x);
		meshData.aabb[0].y = std::min(meshData.aabb[0].y, meshData.vertices[meshData.vertices.size()-1].y);
		meshData.aabb[0].z = std::min(meshData.aabb[0].z, meshData.vertices[i].z);

		meshData.aabb[1].x = std::max(meshData.aabb[1].x, meshData.vertices[i].x);
		meshData.aabb[1].y = std::max(meshData.aabb[1].y, meshData.vertices[i].y);
		meshData.aabb[1].z = std::max(meshData.aabb[1].z, meshData.vertices[i].z);

	}

	for (int i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace face = mesh->mFaces[i];
		vector<int> f;
		for (int j = 0; j < face.mNumIndices; j++)
		{
			f.push_back(face.mIndices[j]);
		}
		meshData.faces.push_back(f);
	}


	//meshData.meshCenter = 0.5f*(aabb_min + aabb_max);

	return meshData;
}