#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#include <source.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION    
#include "stb_image_write.h"


cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;

int OpenCL_init() {
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;
	/* Get Platform and Device Info */
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
	/* Create OpenCL context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	/* Create Command Queue */
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
}

cl_program make_program() {
	cl_program program = NULL;
	cl_int ret;
	const char** g[1];
	g[0] = kernal_source;

	program = clCreateProgramWithSource(context, 1, g,
		NULL, &ret);

	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	if (ret != CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char* log = (char*)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
	}
	return program;
}

cl_kernel make_kernal_by_name(cl_program program, const char* name) {
	cl_int ret;
	return clCreateKernel(program, name, &ret);
}

cl_mem make_buffer(size_t size) {
	cl_uchar fill_value = 0;
	cl_int ret;
	cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
	ret = clEnqueueFillBuffer(command_queue, buffer, &fill_value, sizeof(fill_value), 0, size, 0, NULL, NULL);
	return buffer;
}

void fill_buffer(cl_mem buffer, size_t size) {
	cl_uchar fill_value = 0;
	cl_int ret;
	ret = clEnqueueFillBuffer(command_queue, buffer, &fill_value, sizeof(fill_value), 0, size, 0, NULL, NULL);
}

void set_buffer_data(cl_mem opencl_buffer, unsigned char* data, size_t size) {
	cl_int ret;
	ret = clEnqueueWriteBuffer(command_queue, opencl_buffer, CL_TRUE, 0, size, data, 0, NULL, NULL);
}

void get_buffer_data(cl_mem opencl_buffer, unsigned char* data, size_t size) {
	cl_int ret;
	ret = clEnqueueReadBuffer(command_queue, opencl_buffer, CL_TRUE, 0, size, data, 0, NULL, NULL);
}

void launch_kernel(cl_kernel kernel, size_t* work_group, size_t work_group_count, int** arguments, int* arguments_size, int argument_count) {
	cl_int ret;

	for (int argument_index = 0; argument_index < argument_count; argument_index++) {
		ret = clSetKernelArg(kernel, argument_index, arguments_size[argument_index], (void*)&arguments[argument_index]);
	}

	/* Execute OpenCL Kernel */
	ret = clEnqueueNDRangeKernel(command_queue, kernel, work_group_count, NULL, work_group, NULL, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("compile error \n");
	}
}

void release_buffer(cl_mem opencl_buffer) {
	cl_int ret;
	ret = clReleaseMemObject(opencl_buffer);
}

void release_kernel(cl_kernel kernel) {
	cl_int ret;
	ret = clReleaseKernel(kernel);
}

void release_program(cl_program program) {
	cl_int ret;
	ret = clReleaseProgram(program);
}

void OpenCL_release() {
	cl_int ret;
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
}

int main() {
	int componentCount, width, height;
	unsigned char* image = stbi_load("Zak-van-Biljon-17.jpg", &width, &height, &componentCount, 0);
	size_t image_size = width * height * componentCount;
	size_t imade_double_size = width * height * sizeof(double);
	int radius = 20;

	OpenCL_init();

	cl_mem image_gpu = make_buffer(image_size);
	set_buffer_data(image_gpu, image, image_size);

	cl_mem image_buffer_real = make_buffer(imade_double_size);
	cl_mem image_buffer_imaginary = make_buffer(imade_double_size);
	cl_mem image_buffer_real_temp = make_buffer(imade_double_size);
	cl_mem image_buffer_imaginary_temp = make_buffer(imade_double_size);

	cl_program program = make_program();
	cl_kernel dft = make_kernal_by_name(program, "KernelDFT");
	cl_kernel convertToGrayscale = make_kernal_by_name(program, "KernalConvertToGrayscale");
	cl_kernel circle_mask = make_kernal_by_name(program, "KernalMask");
	cl_kernel dft_to_image = make_kernal_by_name(program, "KernalDFTImage");


	printf("1. Convert to grayscale\n");

	const size_t wirk_group_size = 2;
	const size_t localSize[2] = { width, height };

	int* arguments[10];
	arguments[0] = image_gpu;
	arguments[1] = componentCount;
	arguments[2] = image_buffer_real;

	int arguments_size[10];
	arguments_size[0] = sizeof(cl_mem);
	arguments_size[1] = sizeof(int);
	arguments_size[2] = sizeof(cl_mem);

	launch_kernel(convertToGrayscale, localSize, wirk_group_size, arguments, arguments_size, 3);
	printf("DFT\n");

	arguments[0] = 1;
	arguments[1] = 0;
	arguments[2] = image_buffer_real;
	arguments[3] = image_buffer_imaginary;
	arguments[4] = image_buffer_real_temp;
	arguments[5] = image_buffer_imaginary_temp;
	arguments_size[0] = sizeof(int);
	arguments_size[1] = sizeof(int);
	arguments_size[2] = sizeof(cl_mem);
	arguments_size[3] = sizeof(cl_mem);
	arguments_size[4] = sizeof(cl_mem);
	arguments_size[5] = sizeof(cl_mem);

	launch_kernel(dft, localSize, wirk_group_size, arguments, arguments_size, 6);

	arguments[0] = 1;
	arguments[1] = 1;
	arguments[2] = image_buffer_real_temp;
	arguments[3] = image_buffer_imaginary_temp;
	arguments[4] = image_buffer_real;
	arguments[5] = image_buffer_imaginary;

	arguments_size[0] = sizeof(int);
	arguments_size[1] = sizeof(int);
	arguments_size[2] = sizeof(cl_mem);
	arguments_size[3] = sizeof(cl_mem);
	arguments_size[4] = sizeof(cl_mem);
	arguments_size[5] = sizeof(cl_mem);

	launch_kernel(dft, localSize, wirk_group_size, arguments, arguments_size, 6);

	arguments[0] = radius * radius;
	arguments[1] = image_buffer_real;
	arguments[2] = image_buffer_imaginary;

	arguments_size[0] = sizeof(int);
	arguments_size[1] = sizeof(cl_mem);
	arguments_size[2] = sizeof(cl_mem);

	launch_kernel(circle_mask, localSize, wirk_group_size, arguments, arguments_size, 3);


	arguments[0] = -1;
	arguments[1] = 0;
	arguments[2] = image_buffer_real;
	arguments[3] = image_buffer_imaginary;
	arguments[4] = image_buffer_real_temp;
	arguments[5] = image_buffer_imaginary_temp;

	arguments_size[0] = sizeof(int);
	arguments_size[1] = sizeof(int);
	arguments_size[2] = sizeof(cl_mem);
	arguments_size[3] = sizeof(cl_mem);
	arguments_size[4] = sizeof(cl_mem);
	arguments_size[5] = sizeof(cl_mem);

	launch_kernel(dft, localSize, wirk_group_size, arguments, arguments_size, 6);


	arguments[0] = -1;
	arguments[1] = 1;
	arguments[2] = image_buffer_real_temp;
	arguments[3] = image_buffer_imaginary_temp;
	arguments[4] = image_buffer_real;
	arguments[5] = image_buffer_imaginary;

	arguments_size[0] = sizeof(int);
	arguments_size[1] = sizeof(int);
	arguments_size[2] = sizeof(cl_mem);
	arguments_size[3] = sizeof(cl_mem);
	arguments_size[4] = sizeof(cl_mem);
	arguments_size[5] = sizeof(cl_mem);

	launch_kernel(dft, localSize, wirk_group_size, arguments, arguments_size, 6);

	arguments[0] = image_gpu;
	arguments[1] = componentCount;
	arguments[2] = image_buffer_real;
	arguments[3] = image_buffer_imaginary;

	arguments_size[0] = sizeof(cl_mem);
	arguments_size[1] = sizeof(int);
	arguments_size[2] = sizeof(cl_mem);
	arguments_size[3] = sizeof(cl_mem);

	launch_kernel(dft_to_image, localSize, wirk_group_size, arguments, arguments_size, 4);

	get_buffer_data(image_gpu, image, image_size);

	stbi_write_png("NewImage.bmp", width, height, componentCount, image, 0);

	return 0;
}

