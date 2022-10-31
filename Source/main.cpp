#include <windows.h>
#include "Render.h"
#include "ModelLoader.h"
#include "Model.h"
#include "Camera.h"

Render render;
vector<Model> models;
vector<UniformGrid> scene;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR cmdlineArgs, int nCmdshow)
{
	WNDCLASS wndclass;
	HWND hwnd;
	MSG msg;
	DWORD   dwStyle;
	RECT    windowRect;

	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpszClassName = "Scene Renderer";
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = (WNDPROC)WndProc;
	wndclass.hInstance = GetModuleHandle(NULL);
	wndclass.hbrBackground = NULL;
	wndclass.hIcon = LoadIcon(hInstance, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_HAND);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);

	if (!RegisterClass(&wndclass))
	{
		MessageBox(NULL, TEXT("TEXT INSIDE MESSAGE BOX"), TEXT("TITLE FOR THE MESSAGE BOX"), MB_OK);
		return -1;
	}

	AdjustWindowRectEx(&windowRect, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_APPWINDOW | WS_EX_WINDOWEDGE);

	int width = 1000, height = 1000;

	hwnd = CreateWindowEx(WS_EX_APPWINDOW | WS_EX_WINDOWEDGE,
		"Scene Renderer",
		TEXT("Scene Renderer"),
		WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
		0,
		0,
		width,
		height,
		NULL,
		NULL,
		hInstance,
		NULL);

	ShowWindow(hwnd, SW_SHOW);

	ModelLoader modelLoader;
	modelLoader.loadModel("Stage_objects.obj");


	for (int i = modelLoader.meshes.size()-1; i >= 0; i--)
	{
		UniformGrid uniformGrid;
		if(modelLoader.meshes[i].faces.size() < 50)  uniformGrid = UniformGrid(modelLoader, i, 50);
		else if (modelLoader.meshes[i].faces.size() < 1000) uniformGrid = UniformGrid(modelLoader, i, 50);
		else uniformGrid = UniformGrid(modelLoader, i, 100);
		uniformGrid.addObject();
		scene.push_back(uniformGrid);
	}

	scene[0].model->color = glm::vec3(0.89, 0.89, 0.89);
	scene[0].model->mat = emmissive;
	scene[1].model->color = glm::vec3(0.89, 0, 0.89);
	scene[1].model->mat = diffuse;
	scene[2].model->color = glm::vec3(0.89, 0.89, 0.89);
	scene[2].model->mat = diffuse;
	scene[3].model->color = glm::vec3(0.89, 0.89, 0);
	scene[3].model->mat = diffuse;
	scene[4].model->color = glm::vec3(0.89, 0.89, 0.89);
	scene[4].model->mat = diffuse;
	scene[5].model->color = glm::vec3(0.89, 0, 0.89);
	scene[5].model->mat = diffuse;
	scene[6].model->color = glm::vec3(0.89, 0.89, 0.89);
	scene[6].model->mat = diffuse;
	scene[7].model->color = glm::vec3(0.89, 0.89, 0.89);
	scene[7].model->mat = diffuse;
	scene[8].model->color = glm::vec3(0.89, 0.89, 0.89);
	scene[8].model->mat = reflective;
	//scene[6].model->color = glm::vec3(0.75, 0.2, 0.5);

	render.cam.scene = scene;
	render.cam.preProcess();

	render.display();

	PeekMessage(&msg, hwnd, NULL, NULL, PM_REMOVE);
	while (msg.message != WM_QUIT)
	{
		PeekMessage(&msg, hwnd, NULL, NULL, PM_REMOVE);

		if (msg.message == WM_QUIT)
		{
			break;
		}
		TranslateMessage(&msg);
		DispatchMessage(&msg);

	}
	return 0;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	static HGLRC hRC;
	static HDC hDC;
	int width, height;

	switch (msg)
	{
	case WM_CREATE:
	{
		hDC = GetDC(hwnd);
		render.setDC(hDC);
		render.setupPixelFormat();
		hRC = wglCreateContext(hDC);
		wglMakeCurrent(hDC, hRC);
		render.glCtxInit();
		break;
	}
	case WM_SIZE:
	{
		height = HIWORD(lparam);
		width = LOWORD(lparam);
		break;
	}
	case WM_KEYDOWN:
	{
		switch (LOWORD(wparam))
		{
			case VK_LEFT:
			{
				//render.cam.rotateYaw(LEFT);
				//render.display();
				break;
			}
			case VK_RIGHT:
			{
				//render.cam.rotateYaw(RIGHT);
				//render.display();
				break;
			}
			case VK_UP:
			{
				//render.cam.moveForward();
				//render.display();
				break;
			}
			case VK_DOWN:
			{
				//render.cam.moveBackward();
				//render.display();
				break;
			}
		}
		break;
	}
	case WM_LBUTTONDOWN:
	{
		break;
	}
	case WM_CLOSE:
	{
		wglMakeCurrent(hDC, NULL);
		wglDeleteContext(hRC);
		PostQuitMessage(0);
		DestroyWindow(hwnd);
		break;
	}
	case WM_DESTROY:
	{
		PostQuitMessage(0);
		DestroyWindow(hwnd);
		break;
	}
	default:
		return DefWindowProc(hwnd, msg, wparam, lparam);
	}
}

