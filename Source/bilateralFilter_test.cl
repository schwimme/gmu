/*
* GMU projekt - Bilater�ln� filtr
*
* Auto�i: Tom� Pelka (xpelka01).
*/

/**
* @brief Bilater�ln� filtr - testovac� kernel (pro ov��en� pr�ce s OpenCV).
* 
* Na v�stup zkop�ruje vstupn� obraz bez halo z�n.
*
* @param Vstupn� obrazov� data. Barevn� form�t CIE-LAB.
* @param V�stupn� obrazov� data. Barevn� form�t CIE-LAB. Rozm�ry jsou men�� o 2x radius.
* @param Prostorov� (spatial) parametr filtru - radius.
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

	// Okrajov� ��sti (tzv. halo z�ny) pot�ebujeme pro v�po�et, ale nem��eme je zahrnout do v�sledn�ho obr�zku.
	// int dst_width = get_global_size(0); // POZOR - tohle je �patn�! global_size je toti� zarovn�na!
	int src_width = dst_width + space_param * 2;

	// Kontrola hranic - zarovn�n�.
	if ((global_x < dst_width) && (global_y < dst_height))
	{
		destination[global_y * dst_width + global_x] = source[(global_y + space_param) * src_width + global_x + space_param];
	}
}
