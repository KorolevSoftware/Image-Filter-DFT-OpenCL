__kernel void KernalConvertToGrayscale(__global unsigned char* input_image, int componentCount, __global double* output_image) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	int width = get_global_size(0);
	int height = get_global_size(1);
	int pixel_index_r = x * componentCount + y * width * componentCount;
	double temp =
	0.299 * input_image[pixel_index_r + 0] +
	0.587 * input_image[pixel_index_r + 1] +
	0.114 * input_image[pixel_index_r + 2];

	output_image[y * width + x] = temp;
}

__kernel void KernalMask(int radius_2_pow, __global double* input_image_1, __global double* input_image_2) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	int width = get_global_size(0);
	int height = get_global_size(1);

	int middlex = width/ 2;
	int middley = height/ 2;

	if ((x - middlex) * (x - middlex) + (y - middley) * (y - middley) < radius_2_pow) {
		input_image_1[y * width + x] = 0;
		input_image_2[y * width + x] = 0;
	}
}

__kernel void KernalDFTImage(__global unsigned char* output_image, int componentCount,  __global double* buffer_real,  __global double* buffer_imaginary) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	int width = get_global_size(0);
	int height = get_global_size(1);

	double real = buffer_real[y * width + x];
	double imag = buffer_imaginary[y * width + x];
	double dist = distance(real, imag);

	int pixel_index_r = x * componentCount + y * width * componentCount;

	output_image[pixel_index_r + 0] = dist;
	output_image[pixel_index_r + 1] = dist;
	output_image[pixel_index_r + 2] = dist;
}



__kernel void KernelDFT (int dir, int is_vertical, __global double* input_re, __global double* input_im, __global double* DFT_re, __global double* DFT_im) {
	int x = get_global_id(0);
	int y = get_global_id(1);

	int width = get_global_size(0);
	int height = get_global_size(1);

	int size = width;
	int N = x - (width / 2);

	if (is_vertical == 1) {
		int temp;
		temp = width;
		width = height;
		height = temp;
		N = y - (width / 2);
	}

	double arg = -dir*2.0* 3.141592654*convert_double(N) / convert_double(width);
	double real = 0;
	double imag = 0;
	double cosarg, sinarg;
	int pixel_index;
	for (int k = -width / 2; k < width / 2; k++)
	{
		cosarg = cos(k*arg);
		sinarg = sin(k*arg);

		if (is_vertical == 0) {
			pixel_index = y*size + k + (width / 2);
			real += (input_re[pixel_index] * cosarg - input_im[pixel_index] * sinarg);
			imag += (input_re[pixel_index] * sinarg + input_im[pixel_index] * cosarg);
		} else {
			pixel_index = (k + (width / 2))*size + x;
			real += (input_re[pixel_index] * cosarg - input_im[pixel_index] * sinarg);
			imag += (input_re[pixel_index] * sinarg + input_im[pixel_index] * cosarg);
		}
	}
	if (dir == -1) {
		real = real / (width);
		imag = imag / (width);
	}
	DFT_re[y*size + x] = real;
	DFT_im[y*size + x] = imag;
}