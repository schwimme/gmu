/*
 * GMU projekt - Bilater�ln� filtr
 * 
 * Auto�i: Tom� Pelka (xpelka01), Karol Tro�ka (xtrosk00).
 * 
 * Inspirace: https://github.com/OpenCL
 */

#define POW2(x) ((x) * (x))

/**
 * @brief Bilater�ln� filtr - standardn� metoda.
 * 
 * @param Vstupn� obrazov� data. Barevn� form�t CIE-LAB.
 * @param V�stupn� obrazov� data. Barevn� form�t CIE-LAB. Rozm�ry jsou men�� o 2x radius.
 * @param Prostorov� (spatial) parametr filtru - radius.
 * @param Parametr filtru - intenzita barev.
 */
__kernel void bilateral_filter(
	__global float4 *source,
	__global float4 *destination,
	const int space_param,
	const float range_param)
{
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);

	// Okrajov� ��sti (tzv. halo z�ny) pot�ebujeme pro v�po�et, ale nem��eme je zahrnout do v�sledn�ho obr�zku.
	int dst_width = get_global_size(0);
	int src_width = dst_width + space_param * 2;

	// Prost�edn� (referen�n�) bod.
	float4 center_pix = source[(global_y + n_radius) * src_width + global_x + n_radius];

	float4 sum = 0.0f;
	float4 temp_pix = 0.0f;
	float normalization_term = 0.0f; // Ve vzorci je to k(s).
	float spatial_weight, intensity_weight, total_weight;
	int u, v;
	for (int local_y = -space_param; local_y <= space_param; local_y++)
	{
		for (int local_x = -space_param; local_x <= space_param; local_x++)
		{
			u = global_x + local_x + space_param;
			v = global_y + local_y + space_param;

			// Aktu�ln� bod z "okna".
			temp_pix = source[v * src_width + u];

			// V�po�et prostorov� "bl�zkosti" - funkce f ve vzorci.
			// ToDo Optimalizace - vyhled�vac� tabulka, kter� se spo��t� jen jednou.
			spatial_weight = exp(-0.5f * (POW2(local_x) + POW2(local_y)) / space_param);

			// V�po�et barevn� "bl�zkosti" - funkce g ve vzorci.
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
