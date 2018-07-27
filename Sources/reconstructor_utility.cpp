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
		for (int i = pad; i < limit_rows - pad; ++i) {
			for (int j = pad; j < limit_cols - pad; ++j) {
				mat.values[i * limit_rows + j] = initialValue;
			}
		}
	}

	if (pad > 0) {
		const int limit_rows = mat.rows;
		const int limit_cols = mat.cols;
		for (int i = 0; i < pad; ++i) {
			for (int j = 0; j < limit_cols; ++j) {
				mat.values[i * limit_cols + j] = padValue;
			}
		}
		for (int i = limit_rows - pad - 1; i < limit_rows; ++i) {
			for (int j = 0; j < limit_cols; ++j) {
				mat.values[i * limit_cols + j] = padValue;
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

void matrix_reset(matrix &mat, const float initialValue, const float padValue) {
	const int limit_rows = mat.rows;
	const int limit_cols = mat.cols;
	const int pad = mat.padding;

	for (int i = pad; i < limit_rows - pad; ++i) {
		for (int j = pad; j < limit_cols - pad; ++j) {
			mat.values[i * limit_rows + j] = initialValue;
		}
	}

	for (int i = 0; i < pad; ++i) {
		for (int j = 0; j < limit_cols; ++j) {
			mat.values[i * limit_cols + j] = padValue;
		}
	}
	for (int i = limit_rows - pad - 1; i < limit_rows; ++i) {
		for (int j = 0; j < limit_cols; ++j) {
			mat.values[i * limit_cols + j] = padValue;
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

void matrix_reset_all(matrix &target, const float initialValue) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	for (int i = 0; i < mat_row; ++i) {
		for (int j = 0; j < mat_col; ++j) {
			target.values[i * mat_col + j] = initialValue;
		}
	}
}

void matrix_downsample_half(matrix &target, matrix &input) {
	const unsigned int target_pad = target.padding;
	const unsigned int target_cols = target.cols;
	const unsigned int stride = 2;
	const unsigned int input_pad = input.padding;
	const unsigned int input_cols = input.cols;
	const unsigned int input_rows = input.rows;
	const unsigned int in_max = input_cols * input_rows;

	//#pragma omp parallel for
	for (int i = 0, ty = target_pad; i < input_rows; i += stride, ++ty) {
		for (int j = 0, tx = target_pad; j < input_cols; j += stride, ++tx) {
			const unsigned int upper_left = i * input_cols + j;
			target.values[ty * target_cols + tx] = input.values[upper_left];
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

	//#pragma omp parallel for
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
	//#pragma omp parallel for shared(max)
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col; ++j) {
			float val = target.values[i * row_length + j];
			max = val > max ? val : max;
		}
	}

}

void matrix_upsample_zeros_double(matrix &target, matrix &input) {
	const unsigned int in_pad = input.padding;
	const unsigned int input_cols = input.cols;
	const unsigned int target_cols = target.cols;
	const unsigned int target_pad = target.padding;
	const unsigned int target_limit_cols = target_cols - target_pad;
	const unsigned int target_rows = target.rows;
	const unsigned int target_limit_rows = target_rows - target_pad;
	const unsigned int target_limit = target_cols * target.rows;
	const unsigned int stride = 2;

	//#pragma omp parallel for
	//potential problem area iy =
	for (int iy = in_pad, ty = target_pad; ty < target_limit_rows; ++iy, ty += stride) {
		for (int ix = in_pad, tx = target_pad; tx < target_limit_cols; ++ix, tx += stride) {
			const float input_sample = input.values[iy*input_cols + ix];
			const unsigned int target_dest = ty * target_cols + tx;
			if (target_dest >= target_limit) break;
			target.values[target_dest] = input_sample;
		}
	}
}

void matrix_convolve_padded_to_non(matrix &target, matrix &input, matrix &kernel) {
	const unsigned int input_pad = input.padding;
	const unsigned int input_cols = input.cols;
	const unsigned int input_limit_cols = input.cols-input_pad;
	const unsigned int input_rows = input.rows;
	const unsigned int input_limit_rows = input.rows-input_pad;
	const unsigned int target_cols = target.cols;
	const unsigned int target_pad = 0;
	const unsigned int kernel_row_size = kernel.rows;
	const int kernel_row_start = 0 - kernel_row_size / 2;
	const int kernel_row_stop = -(kernel_row_start - 1);
	const unsigned int kernel_col_size = kernel.cols;
	const int kernel_col_start = 0 - kernel_col_size / 2;
	const int kernel_col_stop = -(kernel_col_start - 1);
	float sum;
	// for each pixel
	//#pragma omp parallel for private(sum)
	for (int i = input_pad, ty = 0; i < input_limit_rows; ++i, ++ty) {
		for (int j = input_pad, tx = 0; j < input_limit_cols; ++j, ++tx) {
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
	//if (input.cols != target.cols || input.rows != target.rows) return;
	//if (kernel.cols != kernel.rows) return;
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
	// for each pixel
	//#pragma omp parallel for private(sum)
	for (int i = 0, ty = target_pad; i < input_rows; ++i, ++ty) {
		for (int j = 0, tx = target_pad; j < input_cols; ++j, ++tx) {
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
			if(accum) target.values[ty * target_cols + tx] += sum;
			else target.values[ty * target_cols + tx] = sum;
		}
	}
}


void from_rgba_array_to_rgb_matrices(GLfloat* array, const unsigned int array_width, const unsigned int array_height, matrix *rgb_mat) {
	const unsigned int mat_pad = rgb_mat[0].padding;
	const unsigned int mat_row = rgb_mat[0].rows;
	const unsigned int mat_col = rgb_mat[0].cols;
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
	const unsigned int target_rows = target.rows;
	const unsigned int target_cols = target.cols;
	const unsigned int toAdd1_cols = toAdd1.cols;
	const unsigned int toAdd2_cols = toAdd2.cols;

	for (int i = 0; i < target_rows; ++i) {
		for (int j = 0; j < target_cols; ++j) {
			target.values[i * target_cols + j] = toAdd1.values[i * toAdd1_cols + j] + toAdd2.values[i * toAdd2_cols + j];
		}
	}

}

void matrix_transfer_matrix(matrix &target, matrix &input) {
	const unsigned int target_row = target.rows;
	const unsigned int target_col = target.cols;

	const unsigned int input_row = input.rows;
	const unsigned int input_col = input.cols;
	//#pragma omp parallel for
	for (int i = 0; i < target_row; ++i) {
		for (int j = 0; j < target_col; ++j) {
			target.values[i * target_col + j] = input.values[i * input_col + j];
		}
	}
}

void matrix_average_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	if (target.rows != toAdd1.rows || target.cols != toAdd1.cols || target.padding != toAdd1.padding) return;
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows - mat_pad;
	const unsigned int mat_col = target.cols - mat_pad;
	const unsigned int row_length = target.cols;
	//#pragma omp parallel for shared(target, toAdd1, toAdd2)
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col; ++j) {
			target.values[i * row_length + j] = (toAdd1.values[i * row_length + j] + toAdd2.values[i * row_length + j]) * 0.5f;;
		}
	}
}

void matrix_mult_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_rows = target.rows;
	const unsigned int mat_cols = target.cols;
	const unsigned int target_cols = target.cols;
	for (int i = 0; i < mat_rows; ++i) {
		for (int j = 0; j < mat_cols; ++j) {
			target.values[i * target_cols + j] = toAdd1.values[i * mat_cols + j] * toAdd2.values[i * mat_cols + j];
		}
	}
}

void matrix_mult(matrix &target, float mult) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	for (int i = 0; i < mat_row; ++i) {
		for (int j = 0; j < mat_col; ++j) {
			target.values[i * mat_col + j] *= mult;
		}
	}
}

void matrix_add(matrix &target, float value) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col; ++j) {
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
	for (int i = mat_pad; i < mat_row; ++i) {
		for (int j = mat_pad; j < mat_col; ++j) {
			sum = target.values[i * row_length + j];
		}
	}
	mean = sum / ((mat_row - mat_pad) * (mat_col - mat_pad));
}

void matrix_sub_matrix(matrix &target, matrix &toAdd1, matrix &toAdd2) {
	const unsigned int mat_pad = target.padding;
	const unsigned int mat_row = target.rows;
	const unsigned int mat_col = target.cols;
	//#pragma omp parallel for shared(target, toAdd1, toAdd2)
	for (int i = 0; i < mat_row; ++i) {
		for (int j = 0; j < mat_col; ++j) {
			target.values[i * mat_col + j] = toAdd1.values[i * mat_col + j] - toAdd2.values[i * mat_col + j];
		}
	}
}

void from_rgb_matrices_to_rgba_array(matrix *rgb_mat, GLfloat* array, const unsigned int array_width, const unsigned int array_height) {
	// is +1 to offset the 1 layer of padding on the input
	const unsigned int mat_pad = rgb_mat[0].padding + 1;
	const unsigned int mat_row = rgb_mat[0].rows; //array_height should be mat_row-2*mat_pad
	const unsigned int mat_col = rgb_mat[0].cols; //array_width should be (mat_col-2*mat_pad) * components (which is 4 for rgba)
	const unsigned int stride = 4;
	const unsigned int array_limit = array_width * stride;
	for (int i = 0; i < array_height; ++i) {
		for (int j = 0, k = 0; j < array_limit; j += stride, ++k) {
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