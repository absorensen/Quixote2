//#include "reconstructor.hpp"
//
//Reconstructor::Reconstructor(unsigned int width, unsigned int height, unsigned int budget, unsigned int format, mode mode) {
//	_mode = mode;
//	if (mode == CONV_PYR_CPU) _conv_pyr_cpu = new ConvPyrCPU(width, height, budget, format);
//}
//
//Reconstructor::~Reconstructor() {
//	if (_mode == CONV_PYR_CPU) delete _conv_pyr_cpu;
//}
//
//void Reconstructor::reconstruct_from_gradients(unsigned int &output_texture, bool laplacian) {
//	if (_mode == CONV_PYR_CPU || _mode == FFT_CPU) {
//
//		//ArrayMultAllValues(_data, _width*_height*_input_components, 0.1f);
//
//		// compute
//		//if (_mode == FFT_CPU) DoFFT(_data);
//		if (_mode == CONV_PYR_CPU) _conv_pyr_cpu->reconstruct_from_gradients(output_texture, laplacian);
//		else; //do_fft();
//	}
//	else if (_mode == CONV_PYR_GPU) {
//
//	}
//	else if (_mode == FFT_GPU) {
//
//	}
//	else {
//		// throw error
//	}
//}
//
//
