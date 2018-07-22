#include "convolutional_pyramid_cpu.hpp"

void ConvPyrCPU::reconstruct_from_gradients(unsigned int &output_texture) {
	const unsigned int type = GL_FLOAT;
	// transfer input texture to matrix
	if (_data) {
		glReadPixels(0, 0, _width, _height, _format, type, _data);
	}
	distribute_input();

	// compute
	compute_laplacian();

	// transfer output to output texture
	interleave_output();
	glBindTexture(GL_TEXTURE_2D, output_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, _format, _width, _height, 0, _format, type, _data);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void ConvPyrCPU::distribute_input() {
	from_rgba_array_to_rgb_matrices(_data, _width, _height, _input_image);
}

void ConvPyrCPU::interleave_output() {
	from_rgb_matrices_to_rgba_array(_output_image, _data, _width, _height);
}

void ConvPyrCPU::compute_laplacian() {
	for (int i = 0; i < 3; ++i) {
		//matrix_convolve(_input_image[i], _output_image[i], lap);
		//matrix_add_matrix(_output_image[i], _output_image[i], _input_image[i]);
		matrix_downsample_half(_halfsample_image[i], _input_image[i]);
	}
	for (int i = 0; i < 3; ++i) {
		matrix_upsample_special_double(_output_image[i], _halfsample_image[i]);
	}
}

ConvPyrCPU::ConvPyrCPU(unsigned int width, unsigned int height, unsigned int budget, unsigned int format) {
	_width = width;
	_height = height;
	_budget = budget;
	_format = format;
	_no_of_components = check_format_length(format);

	init_kernels();
	init_input_output();
}


ConvPyrCPU::~ConvPyrCPU() {
	matrix_deallocate(h1);
	matrix_deallocate(h2);
	matrix_deallocate(g);
	matrix_deallocate(lap);
	for (int i = 0; i < 3; ++i) matrix_deallocate(_input_image[i]);
	for (int i = 0; i < 3; ++i) matrix_deallocate(_output_image[i]);
	if (_data != 0) delete [] _data;
}

void ConvPyrCPU::init_kernels()
{

	if (_budget == 2) {
		const int g_width = 5;
		const int g_height = 5;
		matrix_init(g, g_height, g_width);
		g.values[0 * g_width + 0] = 0.0029f; g.values[0 * g_width + 1] = 0.0132f; g.values[0 * g_width + 2] = 0.0310f; g.values[0 * g_width + 3] = 0.0132f; g.values[0 * g_width + 4] = 0.0029f;
		g.values[1 * g_width + 0] = 0.0132f; g.values[1 * g_width + 1] = 0.0598f; g.values[1 * g_width + 2] = 0.1404f; g.values[1 * g_width + 3] = 0.0598f; g.values[1 * g_width + 4] = 0.0132f;
		g.values[2 * g_width + 0] = 0.0310f; g.values[2 * g_width + 1] = 0.1404f; g.values[2 * g_width + 2] = 0.3296f; g.values[2 * g_width + 3] = 0.1404f; g.values[2 * g_width + 4] = 0.0310f;
		g.values[3 * g_width + 0] = 0.0132f; g.values[3 * g_width + 1] = 0.0598f; g.values[3 * g_width + 2] = 0.1404f; g.values[3 * g_width + 3] = 0.0598f; g.values[3 * g_width + 4] = 0.0132f;
		g.values[4 * g_width + 0] = 0.0029f; g.values[4 * g_width + 1] = 0.0132f; g.values[4 * g_width + 2] = 0.0310f; g.values[4 * g_width + 3] = 0.0132f; g.values[4 * g_width + 4] = 0.0029f;

		const int h_width = 7;
		const int h_height = 7;
		matrix_init(h1, h_height, h_width);
		h1.values[0 * h_width] = 0.0037f; h1.values[0 * h_width + 1] = 0.0160f; h1.values[0 * h_width + 2] = 0.0324f; h1.values[0 * h_width + 3] = 0.0403f; h1.values[0 * h_width + 4] = 0.0324f; h1.values[0 * h_width + 5] = 0.0160f; h1.values[0 * h_width + 6] = 0.0037f;
		h1.values[1 * h_width] = 0.0160f; h1.values[1 * h_width + 1] = 0.0685f; h1.values[1 * h_width + 2] = 0.1388f; h1.values[1 * h_width + 3] = 0.1726f; h1.values[1 * h_width + 4] = 0.1388f; h1.values[1 * h_width + 5] = 0.0685f; h1.values[1 * h_width + 6] = 0.0160f;
		h1.values[2 * h_width] = 0.0324f; h1.values[2 * h_width + 1] = 0.1388f; h1.values[2 * h_width + 2] = 0.2813f; h1.values[2 * h_width + 3] = 0.3497f; h1.values[2 * h_width + 4] = 0.2813f; h1.values[2 * h_width + 5] = 0.1388f; h1.values[2 * h_width + 6] = 0.0324f;
		h1.values[3 * h_width] = 0.0403f; h1.values[3 * h_width + 1] = 0.1726f; h1.values[3 * h_width + 2] = 0.3497f; h1.values[3 * h_width + 3] = 0.4347f; h1.values[3 * h_width + 4] = 0.3497f; h1.values[3 * h_width + 5] = 0.1726f; h1.values[3 * h_width + 6] = 0.0403f;
		h1.values[4 * h_width] = 0.0324f; h1.values[4 * h_width + 1] = 0.1388f; h1.values[4 * h_width + 2] = 0.2813f; h1.values[4 * h_width + 3] = 0.3497f; h1.values[4 * h_width + 4] = 0.2813f; h1.values[4 * h_width + 5] = 0.1388f; h1.values[4 * h_width + 6] = 0.0324f;
		h1.values[5 * h_width] = 0.0160f; h1.values[5 * h_width + 1] = 0.0685f; h1.values[5 * h_width + 2] = 0.1388f; h1.values[5 * h_width + 3] = 0.1726f; h1.values[5 * h_width + 4] = 0.1388f; h1.values[5 * h_width + 5] = 0.0685f; h1.values[5 * h_width + 6] = 0.0160f;
		h1.values[6 * h_width] = 0.0037f; h1.values[6 * h_width + 1] = 0.0160f; h1.values[6 * h_width + 2] = 0.0324f; h1.values[6 * h_width + 3] = 0.0403f; h1.values[6 * h_width + 4] = 0.0324f; h1.values[6 * h_width + 5] = 0.0160f; h1.values[6 * h_width + 6] = 0.0037f;

		matrix_init(h2, h_height, h_width);
		h2.values[0 * h_width] = 0.0019f; h2.values[0 * h_width + 1] = 0.0082f; h2.values[0 * h_width + 2] = 0.0166f; h2.values[0 * h_width + 3] = 0.0206f; h2.values[0 * h_width + 4] = 0.0166f; h2.values[0 * h_width + 5] = 0.0082f; h2.values[0 * h_width + 6] = 0.0019f;
		h2.values[1 * h_width] = 0.0082f; h2.values[1 * h_width + 1] = 0.0350f; h2.values[1 * h_width + 2] = 0.0709f; h2.values[1 * h_width + 3] = 0.0882f; h2.values[1 * h_width + 4] = 0.0709f; h2.values[1 * h_width + 5] = 0.0350f; h2.values[1 * h_width + 6] = 0.0082f;
		h2.values[2 * h_width] = 0.0166f; h2.values[2 * h_width + 1] = 0.0709f; h2.values[2 * h_width + 2] = 0.1437f; h2.values[2 * h_width + 3] = 0.1787f; h2.values[2 * h_width + 4] = 0.1437f; h2.values[2 * h_width + 5] = 0.0709f; h2.values[2 * h_width + 6] = 0.0166f;
		h2.values[3 * h_width] = 0.0206f; h2.values[3 * h_width + 1] = 0.0882f; h2.values[3 * h_width + 2] = 0.1787f; h2.values[3 * h_width + 3] = 0.2222f; h2.values[3 * h_width + 4] = 0.1787f; h2.values[3 * h_width + 5] = 0.0882f; h2.values[3 * h_width + 6] = 0.0206f;
		h2.values[4 * h_width] = 0.0166f; h2.values[4 * h_width + 1] = 0.0709f; h2.values[4 * h_width + 2] = 0.1437f; h2.values[4 * h_width + 3] = 0.1787f; h2.values[4 * h_width + 4] = 0.1437f; h2.values[4 * h_width + 5] = 0.0709f; h2.values[4 * h_width + 6] = 0.0166f;
		h2.values[5 * h_width] = 0.0082f; h2.values[5 * h_width + 1] = 0.0350f; h2.values[5 * h_width + 2] = 0.0709f; h2.values[5 * h_width + 3] = 0.0882f; h2.values[5 * h_width + 4] = 0.0709f; h2.values[5 * h_width + 5] = 0.0350f; h2.values[5 * h_width + 6] = 0.0082f;
		h2.values[6 * h_width] = 0.0019f; h2.values[6 * h_width + 1] = 0.0082f; h2.values[6 * h_width + 2] = 0.0166f; h2.values[6 * h_width + 3] = 0.0206f; h2.values[6 * h_width + 4] = 0.0166f; h2.values[6 * h_width + 5] = 0.0082f; h2.values[6 * h_width + 6] = 0.0019f;
	}
	else if (_budget == 1) {
		const int g_width = 3;
		const int g_height = 3;
		matrix_init(g, g_height, g_width);
		g.values[0 * g_width] = 0.0306f; g.values[0 * g_width + 1] = 0.0957f; g.values[0 * g_width + 2] = 0.0306f;
		g.values[1 * g_width] = 0.0957f; g.values[1 * g_width + 1] = 0.2992f; g.values[1 * g_width + 2] = 0.0957f;
		g.values[2 * g_width] = 0.0306f; g.values[2 * g_width + 1] = 0.1404f; g.values[2 * g_width + 2] = 0.0306f;

		const int h_width = 5;
		const int h_height = 5;
		matrix_init(h1, h_height, h_width);
		h1.values[0 * h_width] = 0.0225f; h1.values[0 * h_width + 1] = 0.0750f; h1.values[0 * h_width + 2] = 0.1050f; h1.values[0 * h_width + 3] = 0.0750f; h1.values[0 * h_width + 4] = 0.0225f;
		h1.values[1 * h_width] = 0.0750f; h1.values[1 * h_width + 1] = 0.2500f; h1.values[1 * h_width + 2] = 0.3500f; h1.values[1 * h_width + 3] = 0.2500f; h1.values[1 * h_width + 4] = 0.0750f;
		h1.values[2 * h_width] = 0.1050f; h1.values[2 * h_width + 1] = 0.3500f; h1.values[2 * h_width + 2] = 0.4900f; h1.values[2 * h_width + 3] = 0.3500f; h1.values[2 * h_width + 4] = 0.1050f;
		h1.values[3 * h_width] = 0.0750f; h1.values[3 * h_width + 1] = 0.2500f; h1.values[3 * h_width + 2] = 0.3500f; h1.values[3 * h_width + 3] = 0.2500f; h1.values[3 * h_width + 4] = 0.0750f;
		h1.values[4 * h_width] = 0.0225f; h1.values[4 * h_width + 1] = 0.0750f; h1.values[4 * h_width + 2] = 0.1050f; h1.values[4 * h_width + 3] = 0.0750f; h1.values[4 * h_width + 4] = 0.0225f;

		matrix_init(h2, h_height, h_width);
		h2.values[0 * h_width] = 0.0225f; h2.values[0 * h_width + 1] = 0.0750f; h2.values[0 * h_width + 2] = 0.1050f; h2.values[0 * h_width + 3] = 0.0750f; h2.values[0 * h_width + 4] = 0.0225f;
		h2.values[1 * h_width] = 0.0750f; h2.values[1 * h_width + 1] = 0.2500f; h2.values[1 * h_width + 2] = 0.3500f; h2.values[1 * h_width + 3] = 0.2500f; h2.values[1 * h_width + 4] = 0.0750f;
		h2.values[2 * h_width] = 0.1050f; h2.values[2 * h_width + 1] = 0.3500f; h2.values[2 * h_width + 2] = 0.4900f; h2.values[2 * h_width + 3] = 0.3500f; h2.values[2 * h_width + 4] = 0.1050f;
		h2.values[3 * h_width] = 0.0750f; h2.values[3 * h_width + 1] = 0.2500f; h2.values[3 * h_width + 2] = 0.3500f; h2.values[3 * h_width + 3] = 0.2500f; h2.values[3 * h_width + 4] = 0.0750f;
		h2.values[4 * h_width] = 0.0225f; h2.values[4 * h_width + 1] = 0.0750f; h2.values[4 * h_width + 2] = 0.1050f; h2.values[4 * h_width + 3] = 0.0750f; h2.values[4 * h_width + 4] = 0.0225f;
	}

	const int lap_width = 3;
	const float sign = -1.0f;
	matrix_init(lap, lap_width, lap_width);
	lap.values[0 * lap_width] = 0.0f;			lap.values[0 * lap_width + 1] = sign * 1.0f;		lap.values[0 * lap_width + 2] = 0.0f;
	lap.values[1 * lap_width] = sign * 1.0f;	lap.values[1 * lap_width + 1] = sign * -4.0f;		lap.values[1 * lap_width + 2] = sign * 1.0f;
	lap.values[2 * lap_width] = 0.0f;			lap.values[2 * lap_width + 1] = sign * 1.0f;		lap.values[2 * lap_width + 2] = 0.0f;
}

void ConvPyrCPU::init_input_output() {
	for (int i = 0; i < 3; ++i) {
		matrix_init(_input_image[i], _height, _width, false, 0.0f, 1, 1.0f);
		matrix_init(_output_image[i], _height, _width, false, 0.0f, 1, 1.0f);
		matrix_init(_halfsample_image[i], _height / 2, _width / 2, false, 0.0f, 1, 1.0f);
	}

	_data = (GLfloat*)malloc(sizeof(GLfloat) * _no_of_components * _width * _height);
}
