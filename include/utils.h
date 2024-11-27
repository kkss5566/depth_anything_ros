#pragma once
#include <opencv2/opencv.hpp>
#include <tuple>

/**
 * @brief Resize image for depth estimation
 * @param img Input image
 * @param width Target width
 * @param height Target height
 * @return Tuple of resized image and original dimensions
 */
std::tuple<cv::Mat, int, int> resize_depth(cv::Mat& img, int width, int height);
