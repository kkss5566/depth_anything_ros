#pragma once

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

class DepthAnything {
public:
    /**
     * @brief Constructor
     * @param model_path Path to the TensorRT engine or ONNX model
     * @param logger TensorRT logger instance
     */
    DepthAnything(std::string model_path, nvinfer1::ILogger &logger);
    
    /**
     * @brief Destructor
     */
    ~DepthAnything();

    /**
     * @brief Predict depth from input image
     * @param image Input RGB image
     * @return CV_32FC1 depth map
     */
    cv::Mat predict(cv::Mat &image);

    /**
     * @brief Create colored visualization of depth map
     * @param depth_map Input depth map (CV_32FC1)
     * @param use_rainbow Whether to use rainbow colormap
     * @return Colored depth map (CV_8UC3)
     */
    cv::Mat visualizeDepth(const cv::Mat& depth_map, bool use_rainbow = true);

    /**
     * @brief Get last inference time in milliseconds
     */
    float getInferenceTime() const { return inference_time_ms_; }

private:
    // TensorRT related members
    nvinfer1::IRuntime* runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};
    
    // CUDA members
    cudaStream_t stream;
    void* buffer[2]{nullptr, nullptr};
    float* depth_data{nullptr};
    
    // Input parameters
    int input_w{518};
    int input_h{518};
    const float mean[3]{123.675f, 116.28f, 103.53f};
    const float std[3]{58.395f, 57.12f, 57.375f};
    
    // Performance monitoring
    float inference_time_ms_{0.0f};
    
    /**
     * @brief Preprocess input image
     */
    std::vector<float> preprocess(cv::Mat &image);
    
    /**
     * @brief Build TensorRT engine from ONNX model
     */
    void build(std::string onnxPath, nvinfer1::ILogger &logger);
    
    /**
     * @brief Save TensorRT engine to file
     */
    bool saveEngine(const std::string &filename);
    
    /**
     * @brief Initialize CUDA resources
     */
    void initializeCUDA();
    
    /**
     * @brief Clean up CUDA resources
     */
    void cleanupCUDA();
    
    /**
     * @brief Check if GPU device is compatible
     */
    bool checkGPUCompatibility();

    // Prevent copying
    DepthAnything(const DepthAnything&) = delete;
    DepthAnything& operator=(const DepthAnything&) = delete;
};
