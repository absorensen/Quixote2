#pragma once
#ifndef CONV_PYR_CPU_H
#define CONV_PYR_CPU_H
#include <iostream>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <glad/glad.h>
#include "reconstructor_utility.hpp"
#include <cmath>

struct layer {
	matrix _a[3];
	matrix _a_conv[3];
	matrix _mid[3];
	matrix _b[3];
	matrix _b_conv[3];
};

class ConvPyrCPU
{
public:
	ConvPyrCPU::ConvPyrCPU(unsigned int width, unsigned int height, unsigned int budget, unsigned int format);
	ConvPyrCPU::~ConvPyrCPU();

	void ConvPyrCPU::reconstruct_from_gradients(unsigned int &output_texture);

private:
	unsigned int _height, _width, _budget, _no_of_components, _format, _levels;
	//unsigned int _input_texture, _output_texture;
	matrix _h1, _h2, _g, _lap;
	layer *_layers;
	GLfloat *_data = 0;

	void ConvPyrCPU::init_kernels();
	void ConvPyrCPU::init_layers();
	void ConvPyrCPU::distribute_input();
	void ConvPyrCPU::interleave_output();
	void ConvPyrCPU::integrate_pyramid();
	void ConvPyrCPU::clean_up();

};





#endif