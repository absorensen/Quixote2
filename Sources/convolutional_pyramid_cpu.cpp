#include "convolutional_pyramid_cpu.hpp"

void ConvPyrCPU::reconstruct_from_gradients(unsigned int &output_texture) {

	const unsigned int type = GL_FLOAT;
	// transfer input texture to matrix
	if (_data) {
		glReadPixels(0, 0, _width, _height, _format, type, _data);
	}
	//#pragma omp parallel
	//{
		distribute_input();
	//	for (int color = 0; color < 3; ++color) {
	//		matrix_mean(_input[color], _orig_mean[color]);
	//	}

	//	 compute
		integrate_pyramid();

	//	 transfer output to output texture
		interleave_output();
	//}

	glBindTexture(GL_TEXTURE_2D, output_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, _format, _width, _height, 0, _format, type, _data);
	glBindTexture(GL_TEXTURE_2D, 0);
	clean_up();
}

void ConvPyrCPU::distribute_input() {
	from_rgba_array_to_rgb_matrices(_data, _width, _height, _layers[0]._a);
	//for (int color = 0; color < 3; ++color) {
	//	propagate_border_values(_layers[0]._a[color]);
	//}
	
	//from_rgba_array_to_rgb_matrices(_data, _width, _height, _input);
	//compute_laplacian();
	//for (int color = 0; color < 3; ++color) {
	//	matrix_transfer_matrix_padded_target(_layers[0]._a[color], _input[color]);
	//}
}

void ConvPyrCPU::interleave_output() {
	//from_rgb_matrices_to_rgba_array(_input, _data, _width, _height);
	from_rgb_matrices_to_rgba_array(_layers[0]._b_conv, _data, _width, _height);
}

void ConvPyrCPU::clean_up() {
	//for (int color = 0; color < 3; ++color) {
	//	matrix_reset_all(_input[color], _init_value);
	//}
	// set main values and padding
	for (int layer = 0; layer < _levels; ++layer) {
		for (int color = 0; color < 3; ++color) {
			matrix_reset_all(_layers[layer]._a[color], _init_value);
			matrix_reset_all(_layers[layer]._a_conv[color], _init_value);
			matrix_reset_all(_layers[layer]._b[color], _init_value);
			matrix_reset_all(_layers[layer]._b_conv[color], _init_value);
		}
	}
}

void ConvPyrCPU::compute_laplacian() {
	for (int color = 0; color < 3; ++color) {
		matrix_convolve(_layers[0]._a[color], _input[color], _lap, true, false);
	}
}

void ConvPyrCPU::integrate_pyramid() {
	float mean_re[3];

	for (int color = 0; color < 3; ++color) {

		// down
		for (int layer = 1; layer < _levels; ++layer) {
			//propagate_border_values(_layers[layer]._a[color]);
			//propagate_border_values(_layers[layer - 1]._a_conv[color]);
			matrix_convolve(_layers[layer - 1]._a_conv[color], _layers[layer - 1]._a[color], _h1, false, false);

			matrix_downsample_half(_layers[layer]._a[color], _layers[layer - 1]._a_conv[color]);
		}

		// middle
		//propagate_border_values(_layers[_levels - 1]._b_conv[color]);
		matrix_convolve(_layers[_levels - 1]._b_conv[color], _layers[_levels - 1]._a[color], _g, false, false);

		// up
		for (int layer = _levels - 2; layer >= 0; --layer) {
			//propagate_border_values(_layers[layer]._b[color]);
			//propagate_border_values(_layers[layer]._b_conv[color]);
			matrix_upsample_zeros_double(_layers[layer]._b[color], _layers[layer + 1]._b_conv[color], _h1.cols);
			matrix_convolve(_layers[layer]._b_conv[color], _layers[layer]._a[color], _g, false, false);
			matrix_convolve(_layers[layer]._b_conv[color], _layers[layer]._b[color], _h2, false, true);
		}
		//matrix_transfer_matrix(_input[color], _layers[0]._b_conv[color]);

		// adjust
		//matrix_mean(_layers[0]._b_conv[color], mean_re[color]);
		//matrix_add(_layers[0]._b_conv[color], -mean_re[color]);
		//matrix_add(_layers[0]._b_conv[color], _orig_mean[color]);
		//matrix_mult(_layers[0]._b_conv[color], -1.0f);
	}
}

ConvPyrCPU::ConvPyrCPU(unsigned int width, unsigned int height, unsigned int budget, unsigned int format) {
	_width = width;
	_height = height;
	_budget = budget;
	_format = format;
	_no_of_components = check_format_length(format);
	_levels = (unsigned int)ceil(log2(_height > _width ? _height : _width));
	_levels = _levels > 0 ? _levels : 2;
	_levels = 5;
	_pad_value = 0.0f;
	_init_value = 0.0f;

	init_kernels();
	init_layers();
}


ConvPyrCPU::~ConvPyrCPU() {
	matrix_deallocate(_h1);
	matrix_deallocate(_h2);
	matrix_deallocate(_g);
	matrix_deallocate(_lap);
	for (int color = 0; color < 3; ++color) {
		matrix_deallocate(_input[color]);
	}
	for (int i = 0; i < _levels; ++i) {
		for (int j = 0; j < 3; ++j) {
			matrix_deallocate(_layers[i]._a[j]);
			matrix_deallocate(_layers[i]._a_conv[j]);
			matrix_deallocate(_layers[i]._b[j]);
			matrix_deallocate(_layers[i]._b_conv[j]);
		}
	}
	if (_data != 0) delete[] _data;
}

void ConvPyrCPU::init_kernels()
{

	if (_budget == 2) {
		const int g_width = 5;
		const int g_height = 5;
		matrix_init(_g, g_height, g_width);
		_g.values[0 * g_width + 0] = 0.0029f; _g.values[0 * g_width + 1] = 0.0132f; _g.values[0 * g_width + 2] = 0.0310f; _g.values[0 * g_width + 3] = 0.0132f; _g.values[0 * g_width + 4] = 0.0029f;
		_g.values[1 * g_width + 0] = 0.0132f; _g.values[1 * g_width + 1] = 0.0598f; _g.values[1 * g_width + 2] = 0.1404f; _g.values[1 * g_width + 3] = 0.0598f; _g.values[1 * g_width + 4] = 0.0132f;
		_g.values[2 * g_width + 0] = 0.0310f; _g.values[2 * g_width + 1] = 0.1404f; _g.values[2 * g_width + 2] = 0.3296f; _g.values[2 * g_width + 3] = 0.1404f; _g.values[2 * g_width + 4] = 0.0310f;
		_g.values[3 * g_width + 0] = 0.0132f; _g.values[3 * g_width + 1] = 0.0598f; _g.values[3 * g_width + 2] = 0.1404f; _g.values[3 * g_width + 3] = 0.0598f; _g.values[3 * g_width + 4] = 0.0132f;
		_g.values[4 * g_width + 0] = 0.0029f; _g.values[4 * g_width + 1] = 0.0132f; _g.values[4 * g_width + 2] = 0.0310f; _g.values[4 * g_width + 3] = 0.0132f; _g.values[4 * g_width + 4] = 0.0029f;

		const int h_width = 7;
		const int h_height = 7;
		matrix_init(_h1, h_height, h_width);
		_h1.values[0 * h_width] = 0.0037f; _h1.values[0 * h_width + 1] = 0.0160f; _h1.values[0 * h_width + 2] = 0.0324f; _h1.values[0 * h_width + 3] = 0.0403f; _h1.values[0 * h_width + 4] = 0.0324f; _h1.values[0 * h_width + 5] = 0.0160f; _h1.values[0 * h_width + 6] = 0.0037f;
		_h1.values[1 * h_width] = 0.0160f; _h1.values[1 * h_width + 1] = 0.0685f; _h1.values[1 * h_width + 2] = 0.1388f; _h1.values[1 * h_width + 3] = 0.1726f; _h1.values[1 * h_width + 4] = 0.1388f; _h1.values[1 * h_width + 5] = 0.0685f; _h1.values[1 * h_width + 6] = 0.0160f;
		_h1.values[2 * h_width] = 0.0324f; _h1.values[2 * h_width + 1] = 0.1388f; _h1.values[2 * h_width + 2] = 0.2813f; _h1.values[2 * h_width + 3] = 0.3497f; _h1.values[2 * h_width + 4] = 0.2813f; _h1.values[2 * h_width + 5] = 0.1388f; _h1.values[2 * h_width + 6] = 0.0324f;
		_h1.values[3 * h_width] = 0.0403f; _h1.values[3 * h_width + 1] = 0.1726f; _h1.values[3 * h_width + 2] = 0.3497f; _h1.values[3 * h_width + 3] = 0.4347f; _h1.values[3 * h_width + 4] = 0.3497f; _h1.values[3 * h_width + 5] = 0.1726f; _h1.values[3 * h_width + 6] = 0.0403f;
		_h1.values[4 * h_width] = 0.0324f; _h1.values[4 * h_width + 1] = 0.1388f; _h1.values[4 * h_width + 2] = 0.2813f; _h1.values[4 * h_width + 3] = 0.3497f; _h1.values[4 * h_width + 4] = 0.2813f; _h1.values[4 * h_width + 5] = 0.1388f; _h1.values[4 * h_width + 6] = 0.0324f;
		_h1.values[5 * h_width] = 0.0160f; _h1.values[5 * h_width + 1] = 0.0685f; _h1.values[5 * h_width + 2] = 0.1388f; _h1.values[5 * h_width + 3] = 0.1726f; _h1.values[5 * h_width + 4] = 0.1388f; _h1.values[5 * h_width + 5] = 0.0685f; _h1.values[5 * h_width + 6] = 0.0160f;
		_h1.values[6 * h_width] = 0.0037f; _h1.values[6 * h_width + 1] = 0.0160f; _h1.values[6 * h_width + 2] = 0.0324f; _h1.values[6 * h_width + 3] = 0.0403f; _h1.values[6 * h_width + 4] = 0.0324f; _h1.values[6 * h_width + 5] = 0.0160f; _h1.values[6 * h_width + 6] = 0.0037f;

		matrix_init(_h2, h_height, h_width);
		_h2.values[0 * h_width] = 0.0019f; _h2.values[0 * h_width + 1] = 0.0082f; _h2.values[0 * h_width + 2] = 0.0166f; _h2.values[0 * h_width + 3] = 0.0206f; _h2.values[0 * h_width + 4] = 0.0166f; _h2.values[0 * h_width + 5] = 0.0082f; _h2.values[0 * h_width + 6] = 0.0019f;
		_h2.values[1 * h_width] = 0.0082f; _h2.values[1 * h_width + 1] = 0.0350f; _h2.values[1 * h_width + 2] = 0.0709f; _h2.values[1 * h_width + 3] = 0.0882f; _h2.values[1 * h_width + 4] = 0.0709f; _h2.values[1 * h_width + 5] = 0.0350f; _h2.values[1 * h_width + 6] = 0.0082f;
		_h2.values[2 * h_width] = 0.0166f; _h2.values[2 * h_width + 1] = 0.0709f; _h2.values[2 * h_width + 2] = 0.1437f; _h2.values[2 * h_width + 3] = 0.1787f; _h2.values[2 * h_width + 4] = 0.1437f; _h2.values[2 * h_width + 5] = 0.0709f; _h2.values[2 * h_width + 6] = 0.0166f;
		_h2.values[3 * h_width] = 0.0206f; _h2.values[3 * h_width + 1] = 0.0882f; _h2.values[3 * h_width + 2] = 0.1787f; _h2.values[3 * h_width + 3] = 0.2222f; _h2.values[3 * h_width + 4] = 0.1787f; _h2.values[3 * h_width + 5] = 0.0882f; _h2.values[3 * h_width + 6] = 0.0206f;
		_h2.values[4 * h_width] = 0.0166f; _h2.values[4 * h_width + 1] = 0.0709f; _h2.values[4 * h_width + 2] = 0.1437f; _h2.values[4 * h_width + 3] = 0.1787f; _h2.values[4 * h_width + 4] = 0.1437f; _h2.values[4 * h_width + 5] = 0.0709f; _h2.values[4 * h_width + 6] = 0.0166f;
		_h2.values[5 * h_width] = 0.0082f; _h2.values[5 * h_width + 1] = 0.0350f; _h2.values[5 * h_width + 2] = 0.0709f; _h2.values[5 * h_width + 3] = 0.0882f; _h2.values[5 * h_width + 4] = 0.0709f; _h2.values[5 * h_width + 5] = 0.0350f; _h2.values[5 * h_width + 6] = 0.0082f;
		_h2.values[6 * h_width] = 0.0019f; _h2.values[6 * h_width + 1] = 0.0082f; _h2.values[6 * h_width + 2] = 0.0166f; _h2.values[6 * h_width + 3] = 0.0206f; _h2.values[6 * h_width + 4] = 0.0166f; _h2.values[6 * h_width + 5] = 0.0082f; _h2.values[6 * h_width + 6] = 0.0019f;
	}
	else if (_budget == 1) {
		const int g_width = 3;
		const int g_height = 3;
		matrix_init(_g, g_height, g_width);
		_g.values[0 * g_width] = 0.0306f; _g.values[0 * g_width + 1] = 0.0957f; _g.values[0 * g_width + 2] = 0.0306f;
		_g.values[1 * g_width] = 0.0957f; _g.values[1 * g_width + 1] = 0.2992f; _g.values[1 * g_width + 2] = 0.0957f;
		_g.values[2 * g_width] = 0.0306f; _g.values[2 * g_width + 1] = 0.1404f; _g.values[2 * g_width + 2] = 0.0306f;

		//_g.values[0 * g_width] = 0.0f; _g.values[0 * g_width + 1] = 0.0f; _g.values[0 * g_width + 2] = 0.0f;
		//_g.values[1 * g_width] = 0.0f; _g.values[1 * g_width + 1] = 1.0f; _g.values[1 * g_width + 2] = 0.0f;
		//_g.values[2 * g_width] = 0.0f; _g.values[2 * g_width + 1] = 0.0f; _g.values[2 * g_width + 2] = 0.0f;

		const int h_width = 5;
		const int h_height = 5;
		matrix_init(_h1, h_height, h_width);
		_h1.values[0 * h_width] = 0.0225f; _h1.values[0 * h_width + 1] = 0.0750f; _h1.values[0 * h_width + 2] = 0.1050f; _h1.values[0 * h_width + 3] = 0.0750f; _h1.values[0 * h_width + 4] = 0.0225f;
		_h1.values[1 * h_width] = 0.0750f; _h1.values[1 * h_width + 1] = 0.2500f; _h1.values[1 * h_width + 2] = 0.3500f; _h1.values[1 * h_width + 3] = 0.2500f; _h1.values[1 * h_width + 4] = 0.0750f;
		_h1.values[2 * h_width] = 0.1050f; _h1.values[2 * h_width + 1] = 0.3500f; _h1.values[2 * h_width + 2] = 0.4900f; _h1.values[2 * h_width + 3] = 0.3500f; _h1.values[2 * h_width + 4] = 0.1050f;
		_h1.values[3 * h_width] = 0.0750f; _h1.values[3 * h_width + 1] = 0.2500f; _h1.values[3 * h_width + 2] = 0.3500f; _h1.values[3 * h_width + 3] = 0.2500f; _h1.values[3 * h_width + 4] = 0.0750f;
		_h1.values[4 * h_width] = 0.0225f; _h1.values[4 * h_width + 1] = 0.0750f; _h1.values[4 * h_width + 2] = 0.1050f; _h1.values[4 * h_width + 3] = 0.0750f; _h1.values[4 * h_width + 4] = 0.0225f;

		matrix_init(_h2, h_height, h_width);
		_h2.values[0 * h_width] = 0.0225f; _h2.values[0 * h_width + 1] = 0.0750f; _h2.values[0 * h_width + 2] = 0.1050f; _h2.values[0 * h_width + 3] = 0.0750f; _h2.values[0 * h_width + 4] = 0.0225f;
		_h2.values[1 * h_width] = 0.0750f; _h2.values[1 * h_width + 1] = 0.2500f; _h2.values[1 * h_width + 2] = 0.3500f; _h2.values[1 * h_width + 3] = 0.2500f; _h2.values[1 * h_width + 4] = 0.0750f;
		_h2.values[2 * h_width] = 0.1050f; _h2.values[2 * h_width + 1] = 0.3500f; _h2.values[2 * h_width + 2] = 0.4900f; _h2.values[2 * h_width + 3] = 0.3500f; _h2.values[2 * h_width + 4] = 0.1050f;
		_h2.values[3 * h_width] = 0.0750f; _h2.values[3 * h_width + 1] = 0.2500f; _h2.values[3 * h_width + 2] = 0.3500f; _h2.values[3 * h_width + 3] = 0.2500f; _h2.values[3 * h_width + 4] = 0.0750f;
		_h2.values[4 * h_width] = 0.0225f; _h2.values[4 * h_width + 1] = 0.0750f; _h2.values[4 * h_width + 2] = 0.1050f; _h2.values[4 * h_width + 3] = 0.0750f; _h2.values[4 * h_width + 4] = 0.0225f;
	}

	const int lap_width = 3;
	const float sign = -1.0f;
	matrix_init(_lap, lap_width, lap_width);
	_lap.values[0 * lap_width] = 0.0f;			_lap.values[0 * lap_width + 1] = sign * 1.0f;		_lap.values[0 * lap_width + 2] = 0.0f;
	_lap.values[1 * lap_width] = sign * 1.0f;	_lap.values[1 * lap_width + 1] = sign * -4.0f;		_lap.values[1 * lap_width + 2] = sign * 1.0f;
	_lap.values[2 * lap_width] = 0.0f;			_lap.values[2 * lap_width + 1] = sign * 1.0f;		_lap.values[2 * lap_width + 2] = 0.0f;

	matrix_init(_lapx0, 1, lap_width);
	_lapx0.values[0] = 1.0f;	_lapx0.values[1] = -2.0f;	_lapx0.values[2] = 1.0f;

	matrix_init(_lapx1, 1, lap_width);
	_lapx1.values[0] = 0.0f;	_lapx1.values[1] = 1.0f;	_lapx1.values[2] = -1.0f;

	matrix_init(_lapy0, lap_width, 1);
	_lapy0.values[0] = 1.0f;	_lapy0.values[1] = -2.0f;	_lapy0.values[2] = 1.0f;

	matrix_init(_lapy1, lap_width, 1);
	_lapy1.values[0] = 0.0f;	_lapy1.values[1] = 1.0f;	_lapy1.values[2] = -1.0f;
}

void ConvPyrCPU::init_layers() {
	_data = (GLfloat*)malloc(sizeof(GLfloat) * _no_of_components * _width * _height);

	_layers = new layer[_levels];
	unsigned int height = _height;
	unsigned int width = _width;
	const unsigned int pad = _h1.cols;
	const bool init_values = true;
	const float init_value = _init_value;
	const int input_pad = 1;
	_pad_value = 0.0f;
	//for (int color = 0; color < 3; ++color) {
	//	matrix_init(_input[color], height, width, init_values, init_value, input_pad, _pad_value);
	//}
	//height += 2 * input_pad;
	//width += 2 * input_pad;
	for (int i = 0; i < _levels; ++i) {
		for (int color = 0; color < 3; ++color) {
			//pad_value = _orig_mean[color];
			matrix_init(_layers[i]._a[color], height, width, init_values, init_value, pad, _pad_value);
			matrix_init(_layers[i]._a_conv[color], height, width, init_values, init_value, pad, _pad_value);
			matrix_init(_layers[i]._b[color], height, width, init_values, init_value, pad, _pad_value);
			matrix_init(_layers[i]._b_conv[color], height, width, init_values, init_value, pad, _pad_value);
		}
		//height = height / 2 + 2 * pad;
		//width = width / 2 + 2 * pad;
		height = _layers[i]._a[0].rows / 2;//(height / 2);
		width = _layers[i]._a[0].cols / 2;//(width / 2);
	}
	//for (int i = 0; i < _levels; ++i) {
	//	for (int color = 0; color < 3; ++color) {
	//		//pad_value = _orig_mean[color];
	//		matrix_init(_layers[i]._a[color], height, width, init_values, init_value, pad, _pad_value);
	//		matrix_init(_layers[i]._a_conv[color], height, width, init_values, init_value, pad, _pad_value);
	//	}
	//	height = _layers[i]._a[0].rows / 2;
	//	width = _layers[i]._a[0].cols / 2;
	//}
	//height = height * 2;
	//width = width * 2;
	//for (int i = _levels-1; i >= 0; --i) {
	//	for (int color = 0; color < 3; ++color) {
	//		matrix_init(_layers[i]._b_conv[color], height, width, init_values, init_value, 0, _pad_value);
	//		matrix_init(_layers[i]._b[color], height, width, init_values, init_value, pad, _pad_value);
	//	}
	//	height = _layers[i]._a[0].rows * 2;
	//	width = _layers[i]._a[0].cols * 2;
	//}
}


