/*
 * GMU projekt - Bilaterální filtr
 * 
 * Autoøi: Tomáš Pelka (xpelka01), Karol Troška (xtrosk00).
 * 
 * Inspirace: https://github.com/OpenCL
 */

#define POW2(x) ((x) * (x))

/**
 * @brief Bilaterální filtr - standardní metoda.
 * 
 * @param Vstupní obrazová data. Barevný formát CIE-LAB.
 * @param Výstupní obrazová data. Barevný formát CIE-LAB. Rozmìry jsou menší o 2x radius.
 * @param Prostorový (spatial) parametr filtru - radius.
 * @param Parametr filtru - intenzita barev.
 */
__kernel void bilateralFilter_basic(
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
		// Prostøední (referenèní) bod.
		float3 center_pix = source[(global_y + space_param) * src_width + global_x + space_param];

		float3 sum = 0.0f;
		float3 temp_pix = 0.0f;
		float normalization_term = 0.0f; // Ve vzorci je to k(s).
		float spatial_weight, intensity_weight, total_weight;
		int u, v;
		for (int local_y = -space_param; local_y <= space_param; local_y++)
		{
			for (int local_x = -space_param; local_x <= space_param; local_x++)
			{
				u = global_x + local_x + space_param;
				v = global_y + local_y + space_param;

				// Aktuální bod z "okna".
				temp_pix = source[v * src_width + u];

				// Výpoèet prostorové "blízkosti" - funkce f ve vzorci.
				spatial_weight = exp(-0.5f * (POW2(local_x) + POW2(local_y)) / space_param);

				// Výpoèet barevné "blízkosti" - funkce g ve vzorci.
				intensity_weight = exp(
					-(POW2(center_pix.x - temp_pix.x) +
						POW2(center_pix.y - temp_pix.y) +
						POW2(center_pix.z - temp_pix.z))
					* range_param);

				total_weight = intensity_weight * spatial_weight;

				sum += temp_pix * total_weight;
				normalization_term += total_weight;
			}
		}

		destination[global_y * dst_width + global_x] = sum / normalization_term;
	}
}
