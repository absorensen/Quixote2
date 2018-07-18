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

#include "fft.hpp";


int bit_reverse(int i, int N) {
	int j = 0;
	while (N = N >> 1)
	{
		j = (j << 1) | (i & 1);
		i = i >> 1;
	}
	return j;
}

// Added by AB Sørensen
unsigned int FFT::integrate_texture(unsigned int input_texture) {
	has_input_tex = glIsTexture(input_texture);
	if (has_input_tex) fft[0] = input_texture;
	redraw_input();
	do_fft();
	return fft[current_fft];
}

// Added by AB Sørensen
FFT::FFT(unsigned int width, unsigned int height, unsigned int input_tex)
{
	//// two parallel FFTs
	current_fft = 0;
	has_input_tex = glIsTexture(input_tex);
	if (has_input_tex) fft[0] = input_tex;
	
	inverse[0] = false;
	inverse[1] = false;

	//// two dimensions
	size[0] = width;
	size[1] = height;
	for (int i = 0; i < 2; ++i)
	{
		unsigned int s = size[i];
		stages[i] = 0;
		while (s = s >> 1)
			++stages[i];
		// source of serious errors. The constants being multiplied are necessary 
		// and need to scale with the height and width
		butterflyI[i] = new float[2 * size[i] * stages[i] * 64];
		butterflyWR[i] = new float[size[i] * stages[i] * 64];
		butterflyWI[i] = new float[size[i] * stages[i] * 64];
		scramblers[i] = new unsigned int[stages[i]];
		real_weights[i] = new unsigned int[stages[i]];
		imag_weights[i] = new unsigned int[stages[i]];

		create_butterfly_tables(i);
		init_textures(i);
		init_display_lists(i);
	}
	init_framebuffer();
	init_shaders();
}


//FFT::FFT(unsigned int width, unsigned int height,
//	unsigned int input_tex, bool inv1, bool inv2)
//	: draw_input(0), current_fft(0), redrawn(false)
//{
//	////test_glew();
//
//	//// two parallel FFTs
//	has_input_tex = glIsTexture(input_tex);
//	if (has_input_tex) fft[0] = input_tex;
//	inverse[0] = inv1;
//	inverse[1] = inv2;
//
//	//// two dimensions
//	size[0] = width;
//	size[1] = height;
//	for (int i = 0; i < 2; ++i)
//	{
//		unsigned int s = size[i];
//		stages[i] = 0;
//		while (s = s >> 1)
//			++stages[i];
//		// source of serious errors. The constants being multiplied are necessary 
//		// and need to scale with the height and width
//		butterflyI[i] = new float[2 * size[i] * stages[i] * 57];
//		butterflyWR[i] = new float[size[i] * stages[i] * 57];
//		butterflyWI[i] = new float[size[i] * stages[i] * 57];
//		scramblers[i] = new unsigned int[stages[i]];
//		real_weights[i] = new unsigned int[stages[i]];
//		imag_weights[i] = new unsigned int[stages[i]];
//		
//		create_butterfly_tables(i);
//		init_textures(i);
//		init_display_lists(i);
//	}
//	init_framebuffer();
//	init_shaders();
//}

FFT::~FFT()
{
	for (int i = 0; i < 2; ++i)
	{
		delete[] butterflyI[i];
		delete[] butterflyWR[i];
		delete[] butterflyWI[i];
		delete[] scramblers[i];
		delete[] real_weights[i];
		delete[] imag_weights[i];
	}
}

void FFT::create_butterfly_tables(int d)
{
	int n = 0; 
	for (unsigned int i = 0; i < stages[d]; ++i)
	{
		int blocks = 1 << (stages[d] - 1 - i);
		int block_inputs = 1 << i;
		for (int j = 0; j < blocks; ++j)
		{
			for (int k = 0; k < block_inputs; ++k)
			{
				int i1 = j * block_inputs * 2 + k;
				int i2 = i1 + block_inputs;
				float j1, j2;
				if (i == 0)
				{
					j1 = static_cast<float>(bit_reverse(i1, size[d]));
					j2 = static_cast<float>(bit_reverse(i2, size[d]));
				}
				else
				{
					j1 = static_cast<float>(i1);
					j2 = static_cast<float>(i2);
				}
				i1 += n;
				i2 += n;
				
				butterflyI[d][2 * i1] = j1;
				butterflyI[d][2 * i1 + 1] = j2;
				butterflyI[d][2 * i2] = j1;
				butterflyI[d][2 * i2 + 1] = j2;

				//// compute weights
				double angle = 2.0 * M_PI * k * blocks / static_cast<float>(size[d]);
				float wr = static_cast<float>(cos(angle));
				float wi = static_cast<float>(-sin(angle));

				butterflyWR[d][i1] = wr;
				butterflyWI[d][i1] = wi;
				butterflyWR[d][i2] = -wr;
				butterflyWI[d][i2] = -wi;
			}
			n += size[d];
		}
	}
}

void FFT::init_texture(unsigned int tex, GLenum iformat, GLenum format, float* data, int d)
{
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, iformat, d == 1 ? 1 : size[0], d == 0 ? 1 : size[1], 0, format, GL_FLOAT, data);
}

void FFT::init_textures(int d)
{
	glGenTextures(stages[d], scramblers[d]);
	glGenTextures(stages[d], real_weights[d]);
	glGenTextures(stages[d], imag_weights[d]);
	for (unsigned int i = 0; i < stages[d]; ++i)
	{
		init_texture(scramblers[d][i], GL_LUMINANCE_ALPHA32F_ARB, GL_LUMINANCE_ALPHA, &butterflyI[d][i * 2 * size[d]], d);
		init_texture(real_weights[d][i], GL_ALPHA32F_ARB, GL_ALPHA, &butterflyWR[d][i*size[d]], d);
		init_texture(imag_weights[d][i], GL_ALPHA32F_ARB, GL_ALPHA, &butterflyWI[d][i*size[d]], d);
	}
}

void FFT::init_framebuffer()
{
	if (has_input_tex)
	{
		glGenTextures(1, &fft[1]);
		init_texture(fft[1], GL_RGBA32F_ARB, GL_RGBA, 0);
	}
	else
	{
		glGenTextures(2, fft);
		init_texture(fft[0], GL_RGBA32F_ARB, GL_RGBA, 0);
		init_texture(fft[1], GL_RGBA32F_ARB, GL_RGBA, 0);
	}

	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_ARB, fft[0], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE_ARB, fft[1], 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FFT::init_shaders()
{
	std::string path("Glitter/Sources/");
	fft_prog = Shader(path + "fft_minimal.vert", path + "fft.frag");
	disp_prog[0] = Shader(path + "fft_position.vert", path + "fft_display1.frag");
	disp_prog[1] = Shader(path + "fft_position.vert", path + "fft_display2.frag");
	input_prog[0] = Shader(path + "fft_input.vert", path + "fft_input1.frag");
	input_prog[1] = Shader(path + "fft_input.vert", path + "fft_input2.frag");
	
	//glUseProgram(fft_prog);
	fft_prog.use();

	//glUniform1i(glGetUniformLocation(fft_prog, "fft"), 0);
	fft_prog.setInt("fft", 0);

	//glUniform1i(glGetUniformLocation(fft_prog, "scrambler"), 1);
	fft_prog.setInt("scrambler", 1);

	//glUniform1i(glGetUniformLocation(fft_prog, "real_weight"), 2);
	fft_prog.setInt("real_weight", 2);

	//glUniform1i(glGetUniformLocation(fft_prog, "imag_weight"), 3);
	fft_prog.setInt("imag_weight", 3);

	//glUniform1i(glGetUniformLocation(fft_prog, "dimension"), 0);
	fft_prog.setInt("dimension", 0);

	//glUniform2i(glGetUniformLocation(fft_prog, "inverse"), inverse[0], inverse[1]);
	fft_prog.setInt2("inverse", inverse[0], inverse[1]);

	for (int i = 0; i < 2; ++i)
	{
		//glUseProgram(disp_prog[i]);
		disp_prog[i].use();
		
		//glUniform1i(glGetUniformLocation(disp_prog[i], "fft"), 0);
		disp_prog[i].setInt("fft", 0);

		//glUniform2f(glGetUniformLocation(disp_prog[i], "size"), static_cast<float>(size[0]), static_cast<float>(size[1]));
		disp_prog[i].setVec2("size", static_cast<float>(size[0]), static_cast<float>(size[1]));

		//glUseProgram(input_prog[i]);
		input_prog[i].use();

		//glUniform1i(glGetUniformLocation(input_prog[i], "fft"), 0);
		input_prog[i].setInt("fft", 0);
	}
	glUseProgram(0);
}

void FFT::init_display_lists(int d)
{
	disp_lists[d] = glGenLists(stages[d]);
	for (unsigned int i = 0; i < stages[d]; ++i)
	{
		glNewList(disp_lists[d] + i, GL_COMPILE);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, scramblers[d][i]);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, real_weights[d][i]);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, imag_weights[d][i]);

		draw_quad();

		glEndList();
	}
}

void FFT::set_projection() const
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, static_cast<double>(size[0]), 0.0, static_cast<double>(size[1]), -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
}

void FFT::draw_quad() const
{
	glBegin(GL_POLYGON);
	glVertex2f(0.0f, 0.0f);
	glVertex2f(static_cast<float>(size[0]), 0.0f);
	glVertex2f(static_cast<float>(size[0]), static_cast<float>(size[1]));
	glVertex2f(0.0f, static_cast<float>(size[1]));
	glEnd();
}

void FFT::do_stage(int d, unsigned int s)
{
	unsigned int render_fft = current_fft > 0 ? 0 : 1;
	glDrawBuffer(GL_COLOR_ATTACHMENT0 + render_fft);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, fft[current_fft]);

	glCallList(disp_lists[d] + s);

	current_fft = render_fft;
}

void FFT::do_fft(int d)
{
	for (unsigned int i = 0; i < stages[d]; ++i)
		do_stage(d, i);
}

void FFT::do_fft()
{
	int vp[4];
	glGetIntegerv(GL_VIEWPORT, vp);
	glViewport(0, 0, size[0], size[1]);

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	if (!redrawn)
	{
		if (!has_input_tex && draw_input)
		{
			glDrawBuffer(GL_COLOR_ATTACHMENT0 + current_fft);
			draw_input();
		}
	}
	else
		redrawn = false;

	//glUseProgram(fft_prog);
	fft_prog.use();
	for (int i = 0; i < 4; ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glEnable(GL_TEXTURE_RECTANGLE_ARB);
	}

	set_projection();

	for (int i = 0; i < 2; ++i)
	{
		//glUniform1i(glGetUniformLocation(fft_prog, "dimension"), i);
		fft_prog.setInt("dimension", i);
		do_fft(i);
	}

	for (int i = 3; i >= 0; --i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glDisable(GL_TEXTURE_RECTANGLE_ARB);
	}
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(vp[0], vp[1], vp[2], vp[3]);
}

void FFT::redraw_input()
{
	if (has_input_tex)
		current_fft = 0;
	else if (draw_input)
	{
		int vp[4];
		glGetIntegerv(GL_VIEWPORT, vp);
		glViewport(0, 0, size[0], size[1]);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glDrawBuffer(GL_COLOR_ATTACHMENT0 + current_fft);
		draw_input();
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glViewport(vp[0], vp[1], vp[2], vp[3]);
		redrawn = true;
	}
}

void FFT::draw_output(float r, float g, float b, int i) const
{
	if (redrawn)
		glUseProgram(i == 2 ? input_prog[1].ID : input_prog[0].ID);
	else
	{
		unsigned int program = i == 2 ? disp_prog[1].ID : disp_prog[0].ID;
		glUseProgram(program);
		glUniform3f(glGetUniformLocation(program, "color"), r, g, b);
	}
	set_projection();

	glEnable(GL_TEXTURE_RECTANGLE_ARB);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, fft[current_fft]);

	draw_quad();

	glDisable(GL_TEXTURE_RECTANGLE_ARB);
	glUseProgram(0);
}

void FFT::invert(int i)
{
	inverse[i - 1] = !inverse[i - 1];
	//glUseProgram(fft_prog);
	fft_prog.use();

	//glUniform2i(glGetUniformLocation(fft_prog, "inverse"), inverse[0], inverse[1]);
	fft_prog.setInt2("inverse", inverse[0], inverse[1]);
	glUseProgram(0);
}
