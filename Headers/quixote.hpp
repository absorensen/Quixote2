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
#include <omp.h>

#include <stb_image.h>

#include "camera.h"
#include "mesh.hpp"
#include "shader.h"
#include "model.hpp"
#include "os_timer.h"
#include "reconstructor.hpp"

enum cam_mode { NO_CAM_MODEL, NAIVE, PHYSICAL };


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void process_input(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
unsigned int loadTexture(char const * path);
void renderCube();
void renderQuad();

float lerp(float a, float b, float f)
{
	return a + f * (b - a);
}

#endif //~ Glitter Header
