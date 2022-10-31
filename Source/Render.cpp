#include "Render.h"

Render::Render() {}

Render::~Render() {}

void Render::glCtxInit()
{
	setupPixelFormat();

	cam = Camera(glm::vec3(0.0, 0.0, -10.0), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

}

void Render::setupPixelFormat()
{
	int nPixelFormat;

	static PIXELFORMATDESCRIPTOR pfd = {
			sizeof(PIXELFORMATDESCRIPTOR),
			1,
			PFD_DRAW_TO_WINDOW |
			PFD_SUPPORT_OPENGL |
			PFD_DOUBLEBUFFER,
			PFD_TYPE_RGBA,
			32,
			0, 0, 0, 0, 0, 0,
			0,
			0,
			0,
			0, 0, 0, 0,
			16,
			0,
			0,
			PFD_MAIN_PLANE,
			0,
			0, 0, 0 };

	nPixelFormat = ChoosePixelFormat(hDC, &pfd);
	SetPixelFormat(hDC, nPixelFormat, &pfd);
}

void Render::display(void)
{
	cam.tracePath();
	GLuint tex = 0;
	Pixel* pixels = (Pixel*)malloc(sizeof(Pixel)*IMG_X * IMG_Y);

	Pixel* temp = pixels;
	for (int y = 0; y < IMG_Y; y++)
	{
		for (int x = 0; x < IMG_X; x++)
		{
			int i = x + y * IMG_X;
			temp->r = cam.image[i].x;
			temp->g = cam.image[i].y;
			temp->b = cam.image[i].z;

			temp++;
		}
	}

	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, IMG_X, IMG_Y, 0, GL_RGB, GL_FLOAT, NULL);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMG_X, IMG_Y, GL_RGB, GL_FLOAT, pixels);

	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-2, 2, -2, 2, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glColor3ub(255, 255, 255);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, tex);
	
	glBegin(GL_QUADS);
	
	glTexCoord2i(0, 0);
	glVertex2i(-1, -1);
	
	glTexCoord2i(1, 0);
	glVertex2i(-1, 1);
	
	glTexCoord2i(1, 1);
	glVertex2i(1, 1);
	
	glTexCoord2i(0, 1);
	glVertex2i(1, -1);
	
	glEnd();

	SwapBuffers(hDC);
}

