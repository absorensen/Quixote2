#include "reconstructor_utility.hpp"

void matrix_set_all_values(matrix& mat, const float value) {
	const unsigned int limit = mat.rows * mat.cols;

	for (int i = 0; i < limit; ++i) {
		mat.values[i] = value;
	}
}

void matrix_init(matrix &mat, const unsigned int rows, const unsigned int cols, const bool init, const float initialValue, const int pad, const float padValue) {
	mat.cols = cols + pad * 2;
	mat.rows = rows + pad * 2;
	mat.padding = pad;
	mat.values = (float*)std::malloc(mat.rows * mat.cols * sizeof(float));

	if (init) {
		const int limit_rows = mat.rows;
		const int limit_cols = mat.cols;
		for (int i = 0; i < limit_rows; ++i) {
			for (int j = 0; j < limit_cols; ++j) {
				mat.values[i * limit_cols + j] = initialValue;
			}
		}
	}

	if (pad > 0) {
		const int limit_rows = mat.rows;
		const int limit_cols = mat.cols;
		for (int i = 0; i < pad; ++i) {
			for (int j = 0; j < limit_cols; ++j) {
				mat.values[i * limit_cols + j] = initialValue;
			}
		}
		for (int i = rows + pad; i < limit_rows; ++i) {
			for (int j = 0; j < limit_cols; ++j) {
				mat.values[i * limit_cols + j] = initialValue;
			}
		}
		const int pad_sides = limit_rows - pad;
		for (int i = pad; i < pad_sides; ++i) {
			for (int j = 0; j < pad; ++j) {
				mat.values[i * limit_cols + j] = padValue;
			}
			for (int k = limit_cols - pad; k < limit_cols; ++k) {
				mat.values[i * limit_cols + k] = padValue;
			}
		}
	}

	return;
}

void matrix_downsample_half(matrix &target, matrix &input) {
	const unsigned int target_pad = target.padding;
	const unsigned int limit_rows = target.rows - target_pad;
	const unsigned int target_row_length = target.cols;
	const unsigned int limit_cols = target_row_length - target_pad;
	const unsigned int stride = 2;
	const unsigned int input_pad = input.padding;
	const unsigned int input_row_length = input.cols;
	
	#pragma omp parallel for
	for (int i = target_pad; i < limit_rows; ++i) {
		for (int j = target_pad; j < limit_cols; ++j) {
			const unsigned int upper_left = stride * i * input_row_length + stride * j + input_pad;
			target.values[i*target_row_length + j + target_pad] = (input.values[upper_left] + input.values[upper_left + 1] + input.values[upper_left + input_row_length] + input.values[upper_left + input_row_length + 1]) * 0.25f;
		}
	}
}

void matrix_upsample_simple_double(matrix &target, matrix &input) {
	const unsigned int in_pad = input.padding;
	const unsigned int limit_rows = input.rows - in_pad;
	const unsigned int row_length = input.cols;
	const unsigned int limit_cols = row_length - in_pad;
	const unsigned int target_pad = target.padding;
	const unsigned int target_row_length = target.cols;
	const unsigned int stride = 2;

	#pragma omp parallel for
	for (int i = 0; i < limit_rows; ++i) {
		for (int j = 0; j < limit_cols; ++j) {
			const float input_sample = input.values[(i+in_pad)*row_length + j + in_pad];
			const unsigned int target_upper_left = ((i*stride)+target_pad) * target_row_length + (j*stride) + target_pad;
			target.values[target_upper_left] = input_sample;
			target.values[target_upper_left + 1] = input_sample;
			target.values[target_upper_left + target_row_length] = input_sample;
			target.values[target_upper_left + target_row_length + 1] = input_sample;
		}
	}
}

void matrix_upsample_zeros_double(matrix &target, matrix &input) {
	const unsigned int in_pad = input.padding;
	const unsigned int limit_rows = input.rows - in_pad;
	const unsigned int row_length = input.cols;
	const unsigned int limit_cols = row_length - in_pad;
	const unsigned int target_pad = target.padding;
	const unsigned int target_row_length = target.cols;
	const unsigned int stride = 2;

	#pragma omp parallel for
	for (int i = 0; i < limit_rows; ++i) {
		for (int j = 0; j < limit_cols; ++j) {
			const float input_sample = input.values[(i + in_pad)*row_length + j + in_pad];
			const unsigned int target_upper_left = ((i*stride) + target_pad) * target_row_length + (j*stride) + target_pad;
			target.values[target_upper_left] = input_sample;
			//target.values[target_upper_left + 1] = 0.0f;
			//target.values[target_upper_left + target_row_length] = 0.0f;
			//target.values[target_upper_left + target_row_length + 1] = 0.0f;
		}
	}
}

// could be optimized further by creating a function or switch for each of the 
// common kernel sizes which removed the two inner for-loops
void matrix_convolve(matrix &target, matrix &input, matrix &kernel) {
	if (input.cols != target.cols || input.rows != target.rows) return;
	if (kernel.cols != kernel.rows) return;
	const unsigned int input_rows = input.rows;
	const unsigned int input_cols = input.cols;
	const unsigned int input_pad = input.padding;
	const unsigned int kernel_size = kernel.rows;
	const int kernel_start = 0 - kernel_size / 2;
	const int kernel_stop = -(kernel_start - 1);
	const unsigned int kernel_cols = kernel.cols;
	float sum;
	// for each pixel
	#pragma omp parallel for private(sum)
	for (int i = input_pad; i < input_rows - input_pad; ++i) {
		for (int j = input_pad; j < input_cols - input_pad; ++j) {
			sum = 0.0f;

			// convolve
			for (int k = kernel_start, k_row = 0; k < kernel_stop; ++k, ++k_row) {

				if (i + k < 0 || i + k >= input_rows) continue;

				for (int l = kernel_start, k_col = 0; l < kernel_stop; ++l, ++k_col) {
					if (j + l < 0 || j + l >= input_cols) continue;

					// add contribution
					sum += input.values[(i + k)* input_cols + (j + l)] * kernel.values[k_row * kernel_cols + k_col];
				}
			}
			target.values[i * input_cols + j] = sum;
		}
	}
}

void from_rgba_array_to_rgb_matrices(GLfloat* array, const unsigned int array_width, const unsigned int array_height, matrix *rgb_mat) {
	const unsigned int mat_pad = rgb_mat[0].padding;
	const unsigned int mat_row = rgb_mat[0].rows; //array_height should be mat_row-2*mat_pad
	const unsigned int mat_col = rgb_mat[0].cols; //array_width should be (mat_col-2*mat_pad) * components (which is 4 for rgba)
	const unsigned int stride = 4;
	const unsigned int array_limit = 4 * array_width;
	int k = 0;
	for (int i = 0; i < array_height; ++i) {
		for (int j = 0, k = 0; j < array_limit; j += stride, ++k) {
			rgb_mat[0].values[(i + mat_pad) * mat_col + k + mat_pad] = array[i * array_limit + j];
			rgb_mat[1].values[(i + mat_pad) * mat_col + k + mat_pad] = array[i * array_limit + j + 1];
			rgb_mat[2].values[(i + mat_pad) * mat_col + k + mat_pad] = array[i * array_limit + j + 2];
		}
	}
}

void matrix_add_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	if (target.rows != toAdd1.rows || target.cols != toAdd1.cols || target.padding != toAdd1.padding) return;
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	#pragma omp parallel for shared(target, toAdd1, toAdd2)
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col - mat_pad; ++j) {
			target.values[i * mat_col + j] = toAdd1.values[i * mat_col + j] + toAdd2.values[i * mat_col + j];
		}
	}
}

void matrix_transfer_matrix(matrix &target, matrix &input) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	#pragma omp parallel for shared(target, input)
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col - mat_pad; ++j) {
			target.values[i * mat_col + j] = input.values[i * mat_col + j];
		}
	}
}

void matrix_average_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	if (target.rows != toAdd1.rows || target.cols != toAdd1.cols || target.padding != toAdd1.padding) return;
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	#pragma omp parallel for shared(target, toAdd1, toAdd2)
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col - mat_pad; ++j) {
			target.values[i * mat_col + j] = (toAdd1.values[i * mat_col + j] + toAdd2.values[i * mat_col + j]) * 0.5f;;
		}
	}
}

void matrix_mult_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	if (target.rows != toAdd1.rows || target.cols != toAdd1.cols || target.padding != toAdd1.padding) return;
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col - mat_pad; ++j) {
			target.values[i * mat_col + j] = toAdd1.values[i * mat_col + j] * toAdd2.values[i * mat_col + j];
		}
	}
}

void matrix_mult(matrix &target, float mult) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col - mat_pad; ++j) {
			target.values[i * mat_col + j] = target.values[i * mat_col + j] * mult;
		}
	}
}

void from_rgb_matrices_to_rgba_array(matrix *rgb_mat, GLfloat* array, const unsigned int array_width, const unsigned int array_height) {
	const unsigned int mat_pad = rgb_mat[0].padding;
	const unsigned int mat_row = rgb_mat[0].rows; //array_height should be mat_row-2*mat_pad
	const unsigned int mat_col = rgb_mat[0].cols; //array_width should be (mat_col-2*mat_pad) * components (which is 4 for rgba)
	const unsigned int stride = 4;
	const unsigned int array_limit = array_width * stride;
	for (int i = 0; i < array_height; ++i) {
		for (int j = 0 , k = 0; j < array_limit; j += stride, ++k) {
			array[i * array_limit + j] = rgb_mat[0].values[(i + mat_pad) * mat_col + k + mat_pad];
			array[i * array_limit + j + 1] = rgb_mat[1].values[(i + mat_pad) * mat_col + k + mat_pad];
			array[i * array_limit + j + 2] = rgb_mat[2].values[(i + mat_pad) * mat_col + k + mat_pad];
			array[i * array_limit + j + 3] = 1.0f;
		}
	}

}

void matrix_deallocate(matrix &mat) {
	if (mat.values != 0) free(mat.values);
}

void array_set_all_values(GLfloat* array, unsigned int length, float value) {
	for (int i = 0; i < length; ++i) {
		array[i] = value;
	}
}

void array_mult_all_values(GLfloat* array, unsigned int length, float value) {
	for (int i = 0; i < length; ++i) {
		array[i] *= value;
	}
}

unsigned int check_format_length(unsigned int format) {
	switch (format) {
	case GL_BGR:
	case GL_RGB:
		return 3;

	case GL_BGRA:
	case GL_RGBA:
		return 4;

	case GL_ALPHA:
	case GL_LUMINANCE:
		return 1;
	}
}