#pragma once
#include <windows.h>
#include <GL/glew.h>
#include "Camera.h"

struct Pixel
{
	float r;
	float g;
	float b;
};

class Render
{
public:
	Render();
	~Render();
	int winH, winW;
	Camera cam;

	void glCtxInit();
	void display();
	void setupPixelFormat();
	void setDC(HDC hDC) { this->hDC = hDC; }

private:
	HDC hDC;
};

