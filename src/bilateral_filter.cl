__kernel void bilateral_filter (__global int *input, __global int *output, int width, int height)
{
  int global_x = (int)get_global_id(0);
  int global_y = (int)get_global_id(1);
}


__kernel void optimized_bilateral_filter (__global int *input, __global int *output, int width, int height)
{
  int global_x = (int)get_global_id(0);
  int global_y = (int)get_global_id(1);
}


