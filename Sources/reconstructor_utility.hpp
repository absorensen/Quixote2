#pragma once
#ifndef RECONSTRUCTOR_UTILITY_H
#define RECONSTRUCTOR_UTILITY_H

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include <stdlib.h>
#include <iostream>

struct matrix {
	unsigned int padding = 0;
	unsigned int rows;
	unsigned int cols;
	float* values = 0;
};

void matrix_set_all_values(matrix& mat, const float value);
void matrix_init(matrix &mat, unsigned int rows, unsigned int cols, bool init = false, float initialValue = 0.0f, int pad = 0, float padValue = 0.0f);
void from_rgba_array_to_rgb_matrices(GLfloat* array, unsigned int array_width, unsigned int array_height, matrix *rgb_mat);
void from_rgb_matrices_to_rgba_array(matrix *rgb_mat, GLfloat* array, const unsigned int array_width, const unsigned int array_height);
void matrix_deallocate(matrix &mat);
void array_set_all_values(GLfloat* array, unsigned int length, float value);
void array_mult_all_values(GLfloat* array, unsigned int length, float value);
unsigned int check_format_length(unsigned int format);
void matrix_convolve(matrix &target, matrix &input, matrix &kernel, bool ignoreTargetPad, bool accum);
void matrix_convolve_padded_to_non(matrix &target, matrix &input, matrix &kernel);
void matrix_add_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2);
void matrix_transfer_matrix(matrix &target, matrix &input);
void matrix_transfer_matrix_padded_target(matrix &target, matrix &input);
void matrix_max(matrix &target, float &max);
void matrix_mean(matrix &target, float &mean);
void matrix_sub_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2);
void matrix_add(matrix &target, float value);
void matrix_reset(matrix &mat, const float initialValue, const float padValue);
void matrix_reset_all(matrix &mat, const float initialValue);
void matrix_mult(matrix &target, float mult);
void matrix_mult_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2);
void matrix_average_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2);
void matrix_downsample_half(matrix &target, matrix &input);
void matrix_upsample_simple_double(matrix &target, matrix &input);
void matrix_upsample_zeros_double(matrix &target, matrix &input);
inline float bilerp(const float x, const float alpha, const float y, const float beta) { return x * alpha + y * beta; }
inline float quadlerp(const float a, const float alpha, const float b, const float beta, const float c, const float gamma, const float d, const float delta) { return a * alpha + b * beta + c * gamma + d * delta; }



#endif