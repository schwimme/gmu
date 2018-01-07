/*
* GMU projekt - Bilaterální filtr
*
* Autoøi: 
*/

int index_3d(int x, int y, int z, int width, int height)
{
	return x * width * height + y * height + z;
}

int index_2d(int x, int y, int width)
{
	return x * width + y;
}

int clamp_(int min, int max, int x)
{
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

float trilinear_interpolation(__global float2 *data, int height, int width, int depth, float y, float x, float z)
{
		const size_t y_index = clamp_(0, height - 1, y);
		const size_t yy_index = clamp_(0, height - 1, y_index + 1);
		const size_t x_index = clamp_(0, width - 1, x);
		const size_t xx_index = clamp_(0, width - 1, x_index + 1);
		const size_t z_index = clamp_(0, depth - 1, z);
		const size_t zz_index = clamp_(0, depth - 1, z_index + 1);
		const float y_alpha = y - y_index;
		const float x_alpha = x - x_index;
		const float z_alpha = z - z_index;

		return
			(1.0 - y_alpha) * (1.0 - x_alpha) * (1.0 - z_alpha) * data[index_3d(y_index, x_index, z_index, height, width)].x +
			(1.0 - y_alpha) * x_alpha       * (1.0 - z_alpha) * data[index_3d(y_index, xx_index, z_index, height, width)].x +
			y_alpha       * (1.0 - x_alpha) * (1.0 - z_alpha) * data[index_3d(yy_index, x_index, z_index, height, width)].x +
			y_alpha       * x_alpha       * (1.0 - z_alpha) * data[index_3d(yy_index, xx_index, z_index, height, width)].x +
			(1.0 - y_alpha) * (1.0 - x_alpha) * z_alpha       * data[index_3d(y_index, x_index, zz_index, height, width)].x +
			(1.0 - y_alpha) * x_alpha       * z_alpha       * data[index_3d(y_index, xx_index, zz_index, height, width)].x +
			y_alpha       * (1.0 - x_alpha) * z_alpha       * data[index_3d(yy_index, x_index, zz_index, height, width)].x +
			y_alpha       * x_alpha       * z_alpha       * data[index_3d(yy_index, xx_index, zz_index, height, width)].x;
}

__kernel void bilateralFilter_optimized(
	__global float *source,
	__global float *destination,
	__global float2 *data_1,
	__global float2 *data_2,
	const float source_min,
	const float source_max,
	const int width,
	const int height,
	const int space_param,
	const float range_param)
{
	int height_position = get_global_id(0);
	int width_position = get_global_id(1);
	
	const int small_height = ((width - 1) / space_param) + 1 + 2 * 2;
	const int small_width = ((height - 1) / space_param) + 1 + 2 * 2;
	const int small_depth = ((source_min - source_max) / range_param) + 1 + 2 * 2;

	// down sample

	const int small_x = (width_position / space_param + 0.5) + 2;
	const int small_y = (height_position / space_param + 0.5) + 2;
	float z = source[index_2d(height_position, width_position, height)] - source_min;
	const int small_z = (z / range_param + 0.5) + 2;

	__global float2* d = &data_1[index_3d(small_y, small_x, small_z, small_height, small_width)];

	// KTTODO - atomic
	d->x += source[index_2d(height_position, width_position, height)];
	d->y += 1.0;

	barrier(CLK_GLOBAL_MEM_FENCE);

	// convolution
	// Global id in 1d:
	int gid = index_2d(height_position, width_position, height);

	// ids requested for convolution:
	int req_ids = small_height * small_width * small_depth;
	int offset[3];

	for (int dim = 0; dim < 3; ++dim)
	{ // dim = 3 stands for x, y, and depth
		const int off = offset[dim];
		for (int ittr = 0; ittr < 2; ++ittr)
		{
			if (gid < req_ids)
			{
				offset[0] = &(data_1[index_3d(1, 0, 0, small_height, small_width)]) - data_1;
				offset[1] = &(data_1[index_3d(0, 1, 0, small_height, small_width)]) - data_1;
				offset[2] = &(data_1[index_3d(0, 0, 1, small_height, small_width)]) - data_1;

				// swap:
				__global float2* tmp = data_1;
				data_1 = data_2;
				data_2 = tmp;

				int z = gid % small_depth;
				int y = (gid / small_depth) % small_height;
				int x = gid / (small_height * small_depth);

				__global float2 *d_ptr = &data_1[index_3d(y, x, 1, small_height, small_width)];
				__global float2 *b_ptr = &data_2[index_3d(y, x, 1, small_height, small_width)];

				float2 b_prev = *(b_ptr - off);
				float2 b_curr = *b_ptr;
				float2 b_next = *(b_ptr + off);

				*d_ptr = (b_prev + b_next + 2.0 * b_curr) / 4.0;
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
		} // ittr
	} // dim


	if (gid < req_ids && data_1[gid].y != 0)
	{
		data_1[gid].x = data_1[gid].x / data_1[gid].y;
	}	

	barrier(CLK_GLOBAL_MEM_FENCE);

	z = source[index_2d(height_position, width_position, height)] - source_min;
	const float px = width_position / space_param + 2;
	const float py = height_position / space_param + 2;
	const float pz = z / range_param + 2;
	destination[index_2d(height_position, width_position, height)] = trilinear_interpolation(data_1, small_height, small_width, small_depth,py, px, pz);
}
