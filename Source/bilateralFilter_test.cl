/*
* GMU projekt - Bilaterální filtr
*
* Autoøi: Tomáš Pelka (xpelka01).
*/

/**
* @brief Bilaterální filtr - testovací kernel (pro ovìøení práce s OpenCV).
* 
* Na výstup zkopíruje vstupní obraz bez halo zón.
*
* @param Vstupní obrazová data. Barevný formát CIE-LAB.
* @param Výstupní obrazová data. Barevný formát CIE-LAB. Rozmìry jsou menší o 2x radius.
* @param Prostorový (spatial) parametr filtru - radius.
* @param Parametr filtru - intenzita barev.
*/
__kernel void bilateralFilter_test(
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

	// Kontrola hranic - zarovnání.
	if ((global_x < dst_width) && (global_y < dst_height))
	{
		destination[global_y * dst_width + global_x] = source[(global_y + space_param) * src_width + global_x + space_param];
	}
}
