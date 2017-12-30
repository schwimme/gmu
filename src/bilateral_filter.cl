__kernel void bilateral_filter (__global uchar *input, __global uchar *output, int width, int height, float euclian_delta, int radius)
{
	int global_x = (int)get_global_id(0);
	int global_y = (int)get_global_id(1);
	
	int size = width * height;

	int index = ((global_x * width) + global_y) * 3;

	output[index] = input[index]; // B
	output[index + 1] = input[index + 1]; // G
	output[index + 2] = input[index + 2]; // R
}


__kernel void optimized_bilateral_filter (__global uchar *input, __global uchar *output, int width, int height, float euclian_delta, int radius)
{
	int global_x = (int)get_global_id(0);
	int global_y = (int)get_global_id(1);
}


