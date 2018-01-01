/*
* GMU projekt - Bilater�ln� filtr
*
* Auto�i: 
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

	// Okrajov� ��sti (tzv. halo z�ny) pot�ebujeme pro v�po�et, ale nem��eme je zahrnout do v�sledn�ho obr�zku.
	// int dst_width = get_global_size(0); // POZOR - tohle je �patn�! global_size je toti� zarovn�na!
	int src_width = dst_width + space_param * 2;

}
