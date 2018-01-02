/*
* GMU projekt - Bilaterální filtr
*
* Autoøi: 
*/

__kernel void bilateralFilter_optimized(
	__global float *source,
	__global float *destination,
	const int width,
	const int height,
	const int space_param,
	const float range_param)
{
//	const size_t padding_xy = 2, padding_z = 2;
//	double src_min, src_max;
//	cv::minMaxLoc(src, &src_min, &src_max);
//
//	const size_t small_height = static_cast<size_t>((height - 1) / sigma_space) + 1 + 2 * padding_xy;
//	const size_t small_width = static_cast<size_t>((width - 1) / sigma_space) + 1 + 2 * padding_xy;
//	const size_t small_depth = static_cast<size_t>((src_max - src_min) / sigma_color) + 1 + 2 * padding_xy;
//
//	int data_size[] = { small_height, small_width, small_depth };
//	cv::Mat data(3, data_size, CV_32FC2);
//	data.setTo(0);
//
//	// down sample
//	for (int y = 0; y < height; ++y) {
//		for (int x = 0; x < width; ++x) {
//			const size_t small_x = static_cast<size_t>(x / sigma_space + 0.5) + padding_xy;
//			const size_t small_y = static_cast<size_t>(y / sigma_space + 0.5) + padding_xy;
//			const float z = src.at<float>(y, x) - src_min;
//			const size_t small_z = static_cast<size_t>(z / sigma_color + 0.5) + padding_z;
//
//			cv::Vec2f v = data.at<cv::Vec2f>(small_y, small_x, small_z);
//			v[0] += src.at<float>(y, x);
//			v[1] += 1.0;
//			data.at<cv::Vec2f>(small_y, small_x, small_z) = v;
//		}
//	}
//
//	// convolution
//	cv::Mat buffer(3, data_size, CV_32FC2);
//	buffer.setTo(0);
//	int offset[3];
//	offset[0] = &(data.at<cv::Vec2f>(1, 0, 0)) - &(data.at<cv::Vec2f>(0, 0, 0));
//	offset[1] = &(data.at<cv::Vec2f>(0, 1, 0)) - &(data.at<cv::Vec2f>(0, 0, 0));
//	offset[2] = &(data.at<cv::Vec2f>(0, 0, 1)) - &(data.at<cv::Vec2f>(0, 0, 0));
//
//	for (int dim = 0; dim < 3; ++dim) { // dim = 3 stands for x, y, and depth
//		const int off = offset[dim];
//		for (int ittr = 0; ittr < 2; ++ittr) {
//			cv::swap(data, buffer);
//
//			for (int y = 1; y < small_height - 1; ++y) {
//				for (int x = 1; x < small_width - 1; ++x) {
//					cv::Vec2f *d_ptr = &(data.at<cv::Vec2f>(y, x, 1));
//					cv::Vec2f *b_ptr = &(buffer.at<cv::Vec2f>(y, x, 1));
//					for (int z = 1; z < small_depth - 1; ++z, ++d_ptr, ++b_ptr) {
//						cv::Vec2f b_prev = *(b_ptr - off), b_curr = *b_ptr, b_next = *(b_ptr + off);
//						*d_ptr = (b_prev + b_next + 2.0 * b_curr) / 4.0;
//					} // z
//				} // x
//			} // y
//
//		} // ittr
//	} // dim
//
//	  // upsample
//
//	for (cv::MatIterator_<cv::Vec2f> d = data.begin<cv::Vec2f>(); d != data.end<cv::Vec2f>(); ++d)
//	{
//		(*d)[0] /= (*d)[1] != 0 ? (*d)[1] : 1;
//	}
//
//	for (int y = 0; y < height; ++y) {
//		for (int x = 0; x < width; ++x) {
//			const float z = src.at<float>(y, x) - src_min;
//			const float px = static_cast<float>(x) / sigma_space + padding_xy;
//			const float py = static_cast<float>(y) / sigma_space + padding_xy;
//			const float pz = static_cast<float>(z) / sigma_color + padding_z;
//			dst.at<float>(y, x) = trilinear_interpolation<cv::Vec2f>(data, py, px, pz)[0];
//		}
//	}
}
