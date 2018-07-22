#pragma once
#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H
#include <iostream>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <glad/glad.h>
#include "convolutional_pyramid_cpu.hpp"


enum mode { CONV_PYR_CPU, CONV_PYR_GPU, FFT_CPU, FFT_GPU };

class Reconstructor
{
public:
	//modes:	0 == Convolutional Pyramid CPU
	//			1 == Convolutional Pyramid GPU
	//			2 == Fast Fourier Transform CPU
	//			3 == Fast Fourier Transform GPU
	Reconstructor::Reconstructor(unsigned int width, unsigned int height, unsigned int budget, unsigned int input_components, mode mode);
	Reconstructor::~Reconstructor();

	void Reconstructor::reconstruct_from_gradients(unsigned int &output_texture);

private:
	unsigned int _mode;
	ConvPyrCPU* _conv_pyr_cpu;
};
#endif
