#include "reconstructor_utility.hpp"

void matrix_convolve_padded_to_non(matrix &target, matrix &input, matrix &kernel) {
	const unsigned int input_pad = input.padding;
	const unsigned int input_cols = input.cols;
	const unsigned int input_limit_cols = input.cols - input_pad;
	const unsigned int input_rows = input.rows;
	const unsigned int input_limit_rows = input.rows - input_pad;
	const unsigned int target_cols = target.cols;
	const unsigned int target_pad = 0;
	const unsigned int kernel_row_size = kernel.rows;
	const int kernel_row_start = 0 - kernel_row_size / 2;
	const int kernel_row_stop = -(kernel_row_start - 1);
	const unsigned int kernel_col_size = kernel.cols;
	const int kernel_col_start = 0 - kernel_col_size / 2;
	const int kernel_col_stop = -(kernel_col_start - 1);
	float sum;
	for (unsigned int i = input_pad, ty = 0; i < input_limit_rows; ++i, ++ty) {
		for (unsigned int j = input_pad, tx = 0; j < input_limit_cols; ++j, ++tx) {
			sum = 0.0f;
			// convolve
			for (int k = kernel_row_start, k_row = 0; k < kernel_row_stop; ++k, ++k_row) {
				if (i + k < 0 || i + k >= input_rows) continue;
				for (int l = kernel_col_start, k_col = 0; l < kernel_col_stop; ++l, ++k_col) {
					if (j + l < 0 || j + l >= input_cols) continue;
					// add contribution
					sum += input.values[(i + k) * input_cols + j + l] * kernel.values[k_row * kernel_col_size + k_col];
				}
			}
			target.values[ty * target_cols + tx] = sum;
		}
	}
}

// could be optimized further by creating a function or switch for each of the 
// common kernel sizes which removed the two inner for-loops
void matrix_convolve(matrix &target, matrix &input, matrix &kernel, bool uneven = false, bool accum = false) {
	const unsigned int input_cols = input.cols;
	const unsigned int input_rows = input.rows;
	const unsigned int target_cols = target.cols;
	const unsigned int target_pad = uneven ? target.padding : 0;
	const unsigned int kernel_row_size = kernel.rows;
	const int kernel_row_start = 0 - kernel_row_size / 2;
	const int kernel_row_stop = -(kernel_row_start - 1);
	const unsigned int kernel_col_size = kernel.cols;
	const int kernel_col_start = 0 - kernel_col_size / 2;
	const int kernel_col_stop = -(kernel_col_start - 1);
	float sum;
	for (unsigned int i = input.padding, ty = target.padding; i < input_rows-input.padding; ++i, ++ty) {
		for (unsigned int j = input.padding, tx = target.padding; j < input_cols-input.padding; ++j, ++tx) {
			sum = 0.0f;
			// convolve
			for (int k = kernel_row_start, k_row = 0; k < kernel_row_stop; ++k, ++k_row) {
				if (k+i < 0 || k+i >= input_rows) continue;

				for (int l = kernel_col_start, k_col = 0; l < kernel_col_stop; ++l, ++k_col) {
					if (j + l < 0 || j + l >= input_cols) continue;
					// add contribution
					sum += input.values[(i + k) * input_cols + j + l] * kernel.values[k_row * kernel_col_size + k_col];
				}
			}
			if (accum) target.values[ty * target_cols + tx] += sum;
			else target.values[ty * target_cols + tx] = sum;
		}
	}
}

void propagate_border_values(matrix &target) {
	const unsigned int pad = target.padding;
	const unsigned int target_cols = target.cols;
	const unsigned int target_rows = target.rows;

	// top rows
	for (unsigned int i = 0; i < pad; ++i) {
		for (unsigned int j = 0; j < target_cols; ++j) {
			target.values[i * target_cols + j] = target.values[pad * target_cols + j];
		}
	}

	// bottom rows
	for (unsigned int i = target_rows - pad - 1; i < target_rows; ++i) {
		for (unsigned int j = 0; j < target_cols; ++j) {
			target.values[i * target_cols + j] = target.values[(target_rows - pad) * target_cols + j];
		}
	}

	// sides
	const unsigned int pad_sides = target_rows - pad;
	for (unsigned int i = pad; i < pad_sides; ++i) {
		// left sides
		float replicate = target.values[i * target_cols + pad];
		for (unsigned int j = 0; j < pad; ++j) {
			target.values[i * target_cols + j] = replicate;
		}

		// right sides
		replicate = target.values[i * target_cols + (target_cols - pad)];
		for (unsigned int k = target_cols - pad; k < target_cols; ++k) {
			target.values[i * target_cols + k] = replicate;
		}
	}
}

void from_rgba_array_to_rgb_matrices(GLfloat* array, const unsigned int array_width, const unsigned int array_height, matrix *rgb_mat) {
	const unsigned int mat_pad = rgb_mat[0].padding;
	const unsigned int mat_row = rgb_mat[0].rows;
	const unsigned int mat_col = rgb_mat[0].cols;
	const unsigned int target_limit = mat_col - mat_pad;
	const unsigned int stride = 4;
	const unsigned int array_limit = 4 * array_width;
	for (unsigned int i = 0, l = mat_pad; i < array_height; ++i, ++l) {
		for (unsigned int j = 0, k = mat_pad; j < array_limit; j += stride, ++k) {
			rgb_mat[0].values[l * mat_col + k] = array[i * array_limit + j];
			rgb_mat[1].values[l * mat_col + k] = array[i * array_limit + j + 1];
			rgb_mat[2].values[l * mat_col + k] = array[i * array_limit + j + 2];
		}
	}
}

void matrix_downsample_half(matrix &target, matrix &input) {
	const unsigned int target_pad = target.padding;
	const unsigned int target_cols = target.cols;
	const unsigned int stride = 2;
	const unsigned int input_cols = input.cols;
	const unsigned int input_rows = input.rows;

	for (unsigned int i = 0, ty = target_pad; i < input_cols; i += stride, ++ty) {
		for (unsigned int j = 0, tx = target_pad; j < input_rows; j += stride, ++tx) {
			target.values[ty * target_cols + tx] = input.values[i * input_cols + j];
		}
	}
}

void matrix_upsample_zeros_double(matrix &target, matrix &input, unsigned int pad) {
	const unsigned int in_pad = input.padding;
	const unsigned int input_cols = input.cols;
	const unsigned int input_limit_cols = input.cols - in_pad;
	const unsigned int input_rows = input.rows;
	const unsigned int input_limit_rows = input.rows - in_pad;
	const unsigned int target_cols = target.cols;
	const unsigned int target_pad = target.padding;
	const unsigned int target_limit_cols = target_cols - pad;
	const unsigned int target_rows = target.rows;
	const unsigned int target_limit_rows = target_rows - pad;
	const unsigned int target_limit = target_cols * target.rows;
	const unsigned int stride = 2;

	for (unsigned int iy = in_pad, ty = 0; iy < input_limit_rows; ++iy, ty += stride) {
		for (unsigned int ix = in_pad, tx = 0; ix < input_limit_cols; ++ix, tx += stride) {
			target.values[ty * target_cols + tx] = input.values[iy*input_cols + ix];
		}
	}
}




void matrix_transfer_matrix(matrix &target, matrix &input) {
	const unsigned int target_row = target.rows;
	const unsigned int target_col = target.cols;
	const unsigned int input_row = input.rows;
	const unsigned int input_col = input.cols;

	for (unsigned int i = 0; i < target_row; ++i) {
		for (unsigned int j = 0; j < target_col; ++j) {
			target.values[i * target_col + j] = input.values[i * input_col + j];
		}
	}
}

void matrix_transfer_matrix_padded_target(matrix &target, matrix &input) {
	const unsigned int input_cols = input.cols;
	const unsigned int input_rows = input.rows;
	const unsigned int target_cols = target.cols;
	const unsigned int target_pad = target.padding;
	
	for (unsigned int i = 0, ty = target_pad; i < input_rows; ++i, ++ty) {
		for (unsigned int j = 0, tx = target_pad; j < input_cols; ++j, ++tx) {
			target.values[ty * target_cols + tx] = input.values[i * input_cols + j];
		}
	}
}



void from_rgb_matrices_to_rgba_array(matrix *rgb_mat, GLfloat* array, const unsigned int array_width, const unsigned int array_height) {
	const unsigned int mat_pad = rgb_mat[0].padding;
	const unsigned int mat_row = rgb_mat[0].rows; //array_height should be mat_row-2*mat_pad
	const unsigned int mat_col = rgb_mat[0].cols; //array_width should be (mat_col-2*mat_pad) * components (which is 4 for rgba)
	const unsigned int stride = 4;
	const unsigned int array_limit = array_width * stride;

	for (unsigned int i = 0, l = mat_pad; i < array_height; ++i, ++l) {
		for (unsigned int j = 0, k = mat_pad; j < array_limit; j += stride, ++k) {
			array[i * array_limit + j] = rgb_mat[0].values[l * mat_col + k];
			array[i * array_limit + j + 1] = rgb_mat[1].values[l * mat_col + k];
			array[i * array_limit + j + 2] = rgb_mat[2].values[l * mat_col + k];
			array[i * array_limit + j + 3] = 1.0f;
		}
	}
}

void matrix_average_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows - mat_pad;
	const unsigned int mat_col = target.cols - mat_pad;
	const unsigned int row_length = target.cols;

	for (unsigned int i = mat_pad; i < mat_row; ++i) {
		for (unsigned int j = mat_pad; j < mat_col; ++j) {
			target.values[i * row_length + j] = (toAdd1.values[i * row_length + j] + toAdd2.values[i * row_length + j]) * 0.5f;;
		}
	}
}

void matrix_add_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	const unsigned int target_rows = target.rows;
	const unsigned int target_cols = target.cols;
	const unsigned int toAdd1_cols = toAdd1.cols;
	const unsigned int toAdd2_cols = toAdd2.cols;

	for (unsigned int i = 0; i < target_rows; ++i) {
		for (unsigned int j = 0; j < target_cols; ++j) {
			target.values[i * target_cols + j] = toAdd1.values[i * toAdd1_cols + j] + toAdd2.values[i * toAdd2_cols + j];
		}
	}

}

void matrix_mult_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_rows = target.rows;
	const unsigned int mat_cols = target.cols;
	const unsigned int target_cols = target.cols;
	for (unsigned int i = 0; i < mat_rows; ++i) {
		for (unsigned int j = 0; j < mat_cols; ++j) {
			target.values[i * target_cols + j] = toAdd1.values[i * mat_cols + j] * toAdd2.values[i * mat_cols + j];
		}
	}
}

void matrix_mult(matrix &target, float mult) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	for (unsigned int i = 0; i < mat_row; ++i) {
		for (unsigned int j = 0; j < mat_col; ++j) {
			target.values[i * mat_col + j] *= mult;
		}
	}
}

void matrix_add(matrix &target, float value) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_limit_row = target.rows - mat_pad;
	const unsigned int mat_limit_col = target.cols - mat_pad;
	const unsigned int mat_col = target.cols;
	for (unsigned int i = mat_pad; i < mat_limit_row; ++i) {
		for (unsigned int j = mat_pad; j < mat_limit_col; ++j) {
			target.values[i * mat_col + j] += value;
		}
	}
}

void matrix_mean(matrix &target, float &mean) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows - mat_pad;
	const unsigned int mat_col = target.cols - mat_pad;
	const unsigned int row_length = target.cols;
	float sum = 0.0f;
	for (unsigned int i = mat_pad; i < mat_row; ++i) {
		for (unsigned int j = mat_pad; j < mat_col; ++j) {
			sum += target.values[i * row_length + j];
		}
	}
	mean = sum / ((mat_row - mat_pad) * (mat_col - mat_pad));
}

void matrix_sub_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;

	for (unsigned int i = 0; i < mat_row; ++i) {
		for (unsigned int j = 0; j < mat_col; ++j) {
			target.values[i * mat_col + j] = toAdd1.values[i * mat_col + j] - toAdd2.values[i * mat_col + j];
		}
	}
}

void matrix_deallocate(matrix &mat) {
	if (mat.values != 0) free(mat.values);
}

void array_set_all_values(GLfloat* array, unsigned int length, float value) {
	for (unsigned int i = 0; i < length; ++i) {
		array[i] = value;
	}
}

void array_mult_all_values(GLfloat* array, unsigned int length, float value) {
	for (unsigned int i = 0; i < length; ++i) {
		array[i] *= value;
	}
}

void matrix_set_all_values(matrix& mat, const float value) {
	const unsigned int limit = mat.rows * mat.cols;

	for (unsigned int i = 0; i < limit; ++i) {
		mat.values[i] = value;
	}
}

void matrix_init(matrix &mat, const unsigned int rows, const unsigned int cols, const bool init, const float initialValue, const unsigned int pad, const float padValue) {
	mat.cols = cols + pad * 2;
	mat.rows = rows + pad * 2;
	mat.padding = pad;
	mat.values = (float*)std::malloc(mat.rows * mat.cols * sizeof(float));

	if (init) {
		const int limit_rows = mat.rows;
		const int limit_cols = mat.cols;
		for (unsigned int i = pad; i < limit_rows - pad; ++i) {
			for (unsigned int j = pad; j < limit_cols - pad; ++j) {
				mat.values[i * limit_rows + j] = initialValue;
			}
		}
	}

	if (pad > 0) {
		const int limit_rows = mat.rows;
		const int limit_cols = mat.cols;
		for (unsigned int i = 0; i < pad; ++i) {
			for (unsigned int j = 0; j < limit_cols; ++j) {
				mat.values[i * limit_cols + j] = padValue;
			}
		}
		for (unsigned int i = limit_rows - pad - 1; i < limit_rows; ++i) {
			for (unsigned int j = 0; j < limit_cols; ++j) {
				mat.values[i * limit_cols + j] = padValue;
			}
		}
		const int pad_sides = limit_rows - pad;
		for (unsigned int i = pad; i < pad_sides; ++i) {
			for (unsigned int j = 0; j < pad; ++j) {
				mat.values[i * limit_cols + j] = padValue;
			}
			for (unsigned int k = limit_cols - pad; k < limit_cols; ++k) {
				mat.values[i * limit_cols + k] = padValue;
			}
		}
	}

	return;
}

void matrix_reset(matrix &mat, const float initialValue, const float padValue) {
	const int limit_rows = mat.rows;
	const int limit_cols = mat.cols;
	const int pad = mat.padding;

	for (unsigned int i = pad; i < limit_rows - pad; ++i) {
		for (unsigned int j = pad; j < limit_cols - pad; ++j) {
			mat.values[i * limit_rows + j] = initialValue;
		}
	}

	for (unsigned int i = 0; i < pad; ++i) {
		for (unsigned int j = 0; j < limit_cols; ++j) {
			mat.values[i * limit_cols + j] = padValue;
		}
	}
	for (unsigned int i = limit_rows - pad - 1; i < limit_rows; ++i) {
		for (unsigned int j = 0; j < limit_cols; ++j) {
			mat.values[i * limit_cols + j] = padValue;
		}
	}
	const unsigned int pad_sides = limit_rows - pad;
	for (unsigned int i = pad; i < pad_sides; ++i) {
		for (unsigned int j = 0; j < pad; ++j) {
			mat.values[i * limit_cols + j] = padValue;
		}
		for (unsigned int k = limit_cols - pad; k < limit_cols; ++k) {
			mat.values[i * limit_cols + k] = padValue;
		}
	}
}

void matrix_reset_all(matrix &target, const float initialValue) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	for (unsigned int i = 0; i < mat_row; ++i) {
		for (unsigned int j = 0; j < mat_col; ++j) {
			target.values[i * mat_col + j] = initialValue;
		}
	}
}


void matrix_upsample_simple_double(matrix &target, matrix &input) {
	const unsigned int in_pad = input.padding;
	const unsigned int limit_rows = input.rows - in_pad;
	const unsigned int input_row_length = input.cols;
	const unsigned int limit_cols = input_row_length - in_pad;
	const unsigned int target_pad = target.padding;
	const unsigned int target_row_length = target.cols;
	const unsigned int stride = 2;

	for (int i = 0; i < limit_rows; ++i) {
		for (int j = 0; j < limit_cols; ++j) {
			const float input_sample = input.values[(i + in_pad)*input_row_length + j + in_pad];
			const unsigned int target_upper_left = ((i*stride) + target_pad) * target_row_length + (j*stride) + target_pad;
			target.values[target_upper_left] = input_sample;
			target.values[target_upper_left + 1] = input_sample;
			target.values[target_upper_left + target_row_length] = input_sample;
			target.values[target_upper_left + target_row_length + 1] = input_sample;
		}
	}
}

void matrix_max(matrix &target, float &max) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows - mat_pad;
	const unsigned int mat_col = target.cols - mat_pad;
	const unsigned int row_length = target.cols;
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col; ++j) {
			const float val = target.values[i * row_length + j];
			max = val > max ? val : max;
		}
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
	return 0;
}