#include "utils.h"

std::tuple<cv::Mat, int, int> resize_depth(cv::Mat &img, int width, int height) {
    int original_width = img.cols;
    int original_height = img.rows;
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    
    return std::make_tuple(resized, original_width, original_height);
}
