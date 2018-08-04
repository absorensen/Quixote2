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
	matrix _a_down[3];
	matrix _b[3];
	matrix _b_conv[3];
	matrix _b_pad[3];
	matrix _b_up[3];
	matrix _b_out[3];
};

class ConvPyrCPU
{
public:
	ConvPyrCPU::ConvPyrCPU(unsigned int width, unsigned int height, unsigned int budget, unsigned int format);
	ConvPyrCPU::~ConvPyrCPU();

	void ConvPyrCPU::reconstruct_from_gradients(unsigned int &output_texture, bool compute_laplacian);

private:
	unsigned int _height, _width, _budget, _no_of_components, _format, _levels, _pad;
	float _max, _pad_value, _init_value;
	float _orig_mean[3];
	//unsigned int _input_texture, _output_texture;
	matrix _h1, _h2, _g, _lap, _lapx0, _lapx1, _lapy0, _lapy1;
	matrix _input[3];
	layer *_layers;
	GLfloat *_data = 0;

	void ConvPyrCPU::deallocate();
	void ConvPyrCPU::init_kernels();
	void ConvPyrCPU::init_layers();
	void ConvPyrCPU::distribute_input();
	void ConvPyrCPU::interleave_output();
	void ConvPyrCPU::integrate_pyramid();
	void ConvPyrCPU::compute_laplacian();
	void ConvPyrCPU::clean_up();

};





#endif