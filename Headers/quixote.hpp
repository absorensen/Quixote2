// based on implementations by 
// Joey de Vries: https://learnopengl.com/code_viewer_gh.php?code=src/5.advanced_lighting/9.ssao/ssao.cpp
// Joshua Senouf: https://github.com/JoshuaSenouf/GLEngine/blob/master/src/renderer/glengine.cpp
// Kevin Fung: https://github.com/Polytonic/Glitter

// Preprocessor Directives
#ifndef QUIXOTE_H
#define QUIXOTE_H
#pragma once

// System Headers
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <btBulletDynamicsCommon.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>
#include <random>
#include <cstdio>
#include <cstdlib>

// Reference: https://github.com/nothings/stb/blob/master/stb_image.h#L4
// To use stb_image, add this in *one* C++ source file.
//     #define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "camera.h"
#include "mesh.hpp"
#include "shader.h"
#include "model.hpp"
#include "fft.hpp"
#include "os_timer.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void process_input(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
unsigned int loadTexture(char const * path);
void renderCube();
void renderQuad();

void draw_fft_source_g();
void draw_fft_source_rb();
void set_projection();

float lerp(float a, float b, float f)
{
	return a + f * (b - a);
}

#endif //~ Glitter Header
