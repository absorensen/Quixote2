#pragma once
#ifndef CONV_PYR_CPU_H
#define CONV_PYR_CPU_H
#include <iostream>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <glad/glad.h>
#include "reconstructor_utility.hpp"

class ConvPyrCPU
{
public:
	ConvPyrCPU::ConvPyrCPU(unsigned int width, unsigned int height, unsigned int budget, unsigned int format);
	ConvPyrCPU::~ConvPyrCPU();

	void ConvPyrCPU::reconstruct_from_gradients(unsigned int &output_texture);

private:
	unsigned int _height, _width, _budget, _no_of_components, _format;
	//unsigned int _input_texture, _output_texture;
	matrix h1, h2, g, lap;
	matrix _input_image[3];
	matrix _output_image[3];
	matrix _halfsample_image[3];
	GLfloat *_data = 0;

	void ConvPyrCPU::init_kernels();
	void ConvPyrCPU::init_input_output();
	void ConvPyrCPU::distribute_input();
	void ConvPyrCPU::interleave_output();
	void ConvPyrCPU::compute_laplacian();

};





#endif