/*
* GMU projekt - Bilaterální filtr
*
* Autoøi: 
*/

__kernel void bilateralFilter_optimized(
	__global float3 *source,
	__global float3 *destination,
	const int dst_width,
	const int dst_height,
	const int space_param,
	const float range_param)
{
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);

	// Okrajové èásti (tzv. halo zóny) potøebujeme pro výpoèet, ale nemùžeme je zahrnout do výsledného obrázku.
	// int dst_width = get_global_size(0); // POZOR - tohle je špatnì! global_size je totiž zarovnána!
	int src_width = dst_width + space_param * 2;

}
