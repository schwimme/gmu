__kernel void matrix_mul_basic(__global int *a, __global int *b, __global int *c, int a_w, int a_h, int b_w)
{
  int global_x = (int)get_global_id(0);
  int global_y = (int)get_global_id(1);

  int b_h = a_w;
  int c_w = b_w;
  int c_h = a_h;

  int result = 0;
  if((global_x < c_w) && (global_y < c_h))
  {
	  for(int i = 0; i < a_w; i++)
		  {
		    result += a[i + global_y * a_w] * b[global_x + i * b_w];
		  }
	  c[global_x + global_y * c_w] = result;
  }
}

__kernel void matrix_mul_local(__global int *a, __global int *b, __global int *c, int a_w, int a_h, int b_w, __local int *tmp_a, __local int *tmp_b)
{
  int global_x = (int)get_global_id(0);
  int global_y = (int)get_global_id(1);
  int local_x = (int)get_local_id(0);
  int local_y = (int)get_local_id(1);
  int local_w = (int)get_local_size(0);
  int local_h = (int)get_local_size(1);

  int b_h = a_w;
  int c_w = b_w;
  int c_h = a_h;
//===========================================================================================  
  /* ======================================================
   * TODO
   * doplnit telo kernelu - s pouzitim sdilene pameti
   * =======================================================
   */
  // kopirovani dat do docasne pameti
  // synchronizace
  // provedeni vypoctu
}




