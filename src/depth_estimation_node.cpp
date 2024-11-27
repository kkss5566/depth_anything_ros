#include <NvInfer.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "depth_anything.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

class DepthEstimationNode {
private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_depth_;
    ros::Publisher pub_colored_depth_;
    
    DepthAnything* depth_model_;
    std::string frame_id_;
    double depth_scale_;
    bool use_rainbow_colormap_;
    
    // 效能監控
    ros::Time last_time_;
    std::queue<double> processing_times_;
    double fps_;
    
    void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
        try {
            ros::Time start_time = ros::Time::now();
            
            // 轉換圖像
            cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
            frame_id_ = msg->header.frame_id;
            
            // 深度估計
            cv::Mat depth_mat = depth_model_->predict(image);
            
            // 縮放深度圖
            depth_mat = depth_mat * depth_scale_;
            
            // 調整大小
            cv::resize(depth_mat, depth_mat, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
            
            // 創建彩色深度圖
            cv::Mat colored_depth = depth_model_->visualizeDepth(depth_mat, use_rainbow_colormap_);
            
            // 計算 FPS
            ros::Time end_time = ros::Time::now();
            double process_time = (end_time - start_time).toSec();
            processing_times_.push(process_time);
            if (processing_times_.size() > 30) {
                processing_times_.pop();
            }
            
            // 計算平均 FPS
            double avg_time = 0;
            std::queue<double> temp = processing_times_;
            while (!temp.empty()) {
                avg_time += temp.front();
                temp.pop();
            }
            avg_time /= processing_times_.size();
            fps_ = 1.0 / avg_time;
            
            // 在彩色深度圖上顯示 FPS
            cv::putText(colored_depth, 
                       cv::format("FPS: %.2f", fps_),
                       cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX,
                       1.0,
                       cv::Scalar(255, 255, 255),
                       2);
            
            // 發布深度圖和彩色深度圖
            sensor_msgs::ImagePtr depth_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", depth_mat).toImageMsg();
            sensor_msgs::ImagePtr colored_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", colored_depth).toImageMsg();
            
            depth_msg->header.stamp = ros::Time::now();
            depth_msg->header.frame_id = frame_id_;
            colored_msg->header = depth_msg->header;
            
            pub_depth_.publish(depth_msg);
            pub_colored_depth_.publish(colored_msg);
            
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

public:
    DepthEstimationNode() : nh_("~"), fps_(0.0) {
        // 獲取參數
        std::string model_path;
        if (!nh_.getParam("model_path", model_path)) {
            ROS_ERROR("Failed to get model path from rosparam!");
            ros::shutdown();
            return;
        }
        
        if (!nh_.getParam("depth_scale", depth_scale_)) {
            ROS_ERROR("Failed to get depth_scale from rosparam!");
            ros::shutdown();
            return;
        }
        
        nh_.param("use_rainbow_colormap", use_rainbow_colormap_, true);
        
        // 初始化模型
        ROS_INFO("Loading model from %s...", model_path.c_str());
        depth_model_ = new DepthAnything(model_path, logger);
        ROS_INFO("Model loaded successfully!");
        
        // 設置訂閱者和發布者
        sub_image_ = nh_.subscribe("input_image", 1, &DepthEstimationNode::imageCallback, this);
        pub_depth_ = nh_.advertise<sensor_msgs::Image>("depth_registered/image_rect", 1);
        pub_colored_depth_ = nh_.advertise<sensor_msgs::Image>("colored_depth", 1);
        
        last_time_ = ros::Time::now();
    }
    
    ~DepthEstimationNode() {
        if (depth_model_) {
            delete depth_model_;
        }
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "depth_estimation");
    DepthEstimationNode node;
    ros::spin();
    return 0;
}
