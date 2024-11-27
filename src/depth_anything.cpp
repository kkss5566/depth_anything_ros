#include "depth_anything.h"
#include <NvOnnxParser.h>
#include "utils.h"

#define CHECK_CUDA(call) do {                                                    \
    cudaError_t status = call;                                                  \
    if (status != cudaSuccess) {                                                \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - "  \
                  << cudaGetErrorString(status) << std::endl;                   \
        throw std::runtime_error("CUDA Error");                                 \
    }                                                                           \
} while(0)

using namespace nvinfer1;

DepthAnything::DepthAnything(std::string model_path, nvinfer1::ILogger &logger) 
    : runtime(nullptr), engine(nullptr), context(nullptr) {
    
    try {
        // 檢查 GPU 相容性
        if (!checkGPUCompatibility()) {
            throw std::runtime_error("GPU not compatible");
        }

        // 初始化 CUDA 資源
        initializeCUDA();

        // 載入模型
        if (model_path.find(".onnx") == std::string::npos) {
            // 載入 TensorRT engine
            std::ifstream engineStream(model_path, std::ios::binary);
            if (!engineStream.good()) {
                throw std::runtime_error("Failed to open engine file: " + model_path);
            }

            engineStream.seekg(0, std::ios::end);
            const size_t modelSize = engineStream.tellg();
            engineStream.seekg(0, std::ios::beg);
            
            std::vector<char> engineData(modelSize);
            engineStream.read(engineData.data(), modelSize);
            engineStream.close();

            runtime = createInferRuntime(logger);
            engine = runtime->deserializeCudaEngine(engineData.data(), modelSize);
            context = engine->createExecutionContext();
        } else {
            // 從 ONNX 建立 engine
            build(model_path, logger);
            saveEngine(model_path);
        }

        // 獲取輸入維度
#if NV_TENSORRT_MAJOR < 10
        auto input_dims = engine->getBindingDimensions(0);
        input_h = input_dims.d[2];
        input_w = input_dims.d[3];
#else
        auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
        input_h = input_dims.d[2];
        input_w = input_dims.d[3];
#endif

        // 分配 CUDA 記憶體
        CHECK_CUDA(cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&buffer[1], input_h * input_w * sizeof(float)));
        depth_data = new float[input_h * input_w];

    } catch (const std::exception& e) {
        cleanupCUDA();
        throw;
    }
}

DepthAnything::~DepthAnything() {
    cleanupCUDA();
}

void DepthAnything::initializeCUDA() {
    CHECK_CUDA(cudaStreamCreate(&stream));
}

void DepthAnything::cleanupCUDA() {
    if (depth_data) {
        delete[] depth_data;
        depth_data = nullptr;
    }

    if (buffer[0]) {
        CHECK_CUDA(cudaFree(buffer[0]));
        buffer[0] = nullptr;
    }

    if (buffer[1]) {
        CHECK_CUDA(cudaFree(buffer[1]));
        buffer[1] = nullptr;
    }

    if (stream) {
        CHECK_CUDA(cudaStreamDestroy(stream));
    }

    if (context) {
        delete context;
        context = nullptr;
    }

    if (engine) {
        delete engine;
        engine = nullptr;
    }

    if (runtime) {
        delete runtime;
        runtime = nullptr;
    }
}

bool DepthAnything::checkGPUCompatibility() {
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) return false;

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    return prop.major >= 6;  // 確保支援 FP16
}

std::vector<float> DepthAnything::preprocess(cv::Mat &image) {
    std::tuple<cv::Mat, int, int> resized = resize_depth(image, input_w, input_h);
    cv::Mat resized_image = std::get<0>(resized);
    
    std::vector<float> input_tensor;
    input_tensor.reserve(3 * input_h * input_w);
    
    // 使用指針直接訪問像素數據以提高效能
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < input_h; ++i) {
            for (int j = 0; j < input_w; ++j) {
                float pixel = static_cast<float>(resized_image.at<cv::Vec3b>(i, j)[c]);
                input_tensor.push_back((pixel - mean[c]) / std[c]);
            }
        }
    }
    
    return input_tensor;
}

cv::Mat DepthAnything::predict(cv::Mat &image) {
    auto start_time = std::chrono::steady_clock::now();

    cv::Mat clone_image;
    image.copyTo(clone_image);

    // 預處理
    std::vector<float> input = preprocess(clone_image);
    CHECK_CUDA(cudaMemcpyAsync(buffer[0], input.data(),
                              3 * input_h * input_w * sizeof(float),
                              cudaMemcpyHostToDevice, stream));

    // 推理
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2(buffer, stream, nullptr);
#else
    context->executeV2(buffer);
#endif

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 後處理
    CHECK_CUDA(cudaMemcpyAsync(depth_data, buffer[1],
                              input_h * input_w * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));

    auto end_time = std::chrono::steady_clock::now();
    inference_time_ms_ = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // 轉換為 OpenCV Mat
    cv::Mat depth_mat(input_h, input_w, CV_32FC1, depth_data);
    return depth_mat.clone();  // 返回副本以避免記憶體問題
}

cv::Mat DepthAnything::visualizeDepth(const cv::Mat& depth_map, bool use_rainbow) {
    cv::Mat colored;
    cv::Mat normalized;
    
    // 正規化到 0-1
    cv::normalize(depth_map, normalized, 0, 1, cv::NORM_MINMAX, CV_32F);
    
    // 轉換到 0-255
    normalized.convertTo(normalized, CV_8U, 255.0);
    
    if (use_rainbow) {
        cv::applyColorMap(normalized, colored, cv::COLORMAP_RAINBOW);
    } else {
        cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);
    }
    
    return colored;
}

void DepthAnything::build(std::string onnxPath, nvinfer1::ILogger &logger) {
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    
    auto config = builder->createBuilderConfig();
    config->setFlag(BuilderFlag::kFP16);
    
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxPath.c_str(),
                              static_cast<int>(ILogger::Severity::kINFO))) {
        throw std::runtime_error("Failed to parse ONNX model");
    }

    auto plan = builder->buildSerializedNetwork(*network, *config);
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    context = engine->createExecutionContext();

    // 清理
    delete network;
    delete config;
    delete parser;
    delete builder;
}

bool DepthAnything::saveEngine(const std::string &onnxpath) {
    std::string engine_path = onnxpath.substr(0, onnxpath.find_last_of(".")) + ".engine";
    
    if (!engine) return false;

    auto serialized_engine = engine->serialize();
    std::ofstream file(engine_path, std::ios::binary);
    if (!file) return false;

    file.write(static_cast<const char*>(serialized_engine->data()),
               serialized_engine->size());
    
    delete serialized_engine;
    return true;
}
