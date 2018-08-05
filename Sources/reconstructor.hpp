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
	Reconstructor::Reconstructor(unsigned int width, unsigned int height, unsigned int budget, unsigned int format, mode mode) {
		_mode = mode;
		if (mode == CONV_PYR_CPU) _conv_pyr_cpu = new ConvPyrCPU(width, height, budget, format);
	}
	Reconstructor::~Reconstructor() {
		if (_mode == CONV_PYR_CPU) delete _conv_pyr_cpu;
	}

	void Reconstructor::reconstruct_from_gradients(unsigned int &output_texture) {
		if (_mode == CONV_PYR_CPU || _mode == FFT_CPU) {

			if (_mode == CONV_PYR_CPU) _conv_pyr_cpu->reconstruct_from_gradients(output_texture);
			else; 
		}
		else if (_mode == CONV_PYR_GPU) {

		}
		else if (_mode == FFT_GPU) {

		}
		else {
			// throw error
		}
	}

private:
	unsigned int _mode;
	ConvPyrCPU* _conv_pyr_cpu;
};
#endif
