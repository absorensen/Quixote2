// based on an implementation by Jeppe Revall Frisvad
// url: http://www2.compute.dtu.dk/pubdb/views/publication_details.php?id=5771

// FFT class which computes two 2D fast Fourier transforms in parallel
// on the GPU.
//
// Assumes that GLEW is available and has been initialized.
//
// Implementation is based on T. Sumanaweera and D. Liu. Medical Image 
// Reconstruction with the FFT. In GPU Gems 2: Programming Techniques 
// for High-Performance Graphics and General-Purpose Computation, 
// Chapter 48, pp. 765-784, Addison-Wesley, 2005.
//
// Input image is provided as a texture or a callback function which
// draws the input.
//
// Code written by Jeppe Revall Frisvad, 2009
// Copyright (c) DTU Informatics 2009

#ifndef FFT_H
#define FFT_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "shader.h";
//#include <glad/glad.h>
//#include <glm/glm.hpp>
//#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>    

int bit_reverse(int i, int N);
class FFT
{
public:

	unsigned int integrate_texture(unsigned int input_texture);
	FFT::FFT(unsigned int width, unsigned int height, unsigned int input_tex);


	//FFT::FFT(unsigned int width, unsigned int height,
	//	unsigned int input_tex = 0, bool inverse1 = false, bool inverse2 = false);
	FFT::~FFT();

	void set_input(void(*callback)()) { draw_input = callback; }
	void invert(int i = 1);
	void do_fft();
	unsigned int get_output() const { return fft[current_fft]; }
	void draw_output(float r, float g, float b, int i = 1) const;
	void redraw_input();
	bool newly_redrawn() const { return redrawn; }

private:
	bool redrawn;
	bool inverse[2];
	unsigned short has_input_tex;  // GLboolean
	unsigned int size[2];
	unsigned int stages[2];
	float* butterflyI[2];
	float* butterflyWR[2];
	float* butterflyWI[2];
	unsigned int* scramblers[2];
	unsigned int* real_weights[2];
	unsigned int* imag_weights[2];
	unsigned int disp_lists[2];
	unsigned int fft[2];
	unsigned int current_fft;
	unsigned int fbo;
	Shader fft_prog;
	Shader input_prog[2];
	Shader disp_prog[2];
	void(*draw_input)();

	//void test_glew() const;
	void create_butterfly_tables(int d);
	void init_texture(unsigned int tex, GLenum iformat, GLenum format, float* data, int d = 2);
	void init_textures(int d);
	void init_display_lists(int d);
	void init_framebuffer();
	void init_shaders();
	void set_projection() const;
	void draw_quad() const;
	void do_stage(int d, unsigned int s);
	void do_fft(int d);
};

#endif // FFT_H