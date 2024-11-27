#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

class DepthVisualizer
{
public:
    DepthVisualizer() : nh_("~"), viewer_("Depth Anything Visualizer")
    {
        // 初始化參數
        nh_.param("depth_scale", depth_scale_, 0.5f);
        nh_.param("max_depth", max_depth_, 4.0f);
        nh_.param("x_width", x_width_, 500);
        nh_.param("z_height", z_height_, 300);
        nh_.param("sample_step", sample_step_, 4);
        nh_.param("fx", fx_, 500.0f);
        nh_.param("fy", fy_, 500.0f);
        nh_.param("cx", cx_, 320.0f);
        nh_.param("cy", cy_, 240.0f);

        // 創建訂閱者
        sub_ = nh_.subscribe("/depth_anything/depth_registered/image_rect", 1, 
                            &DepthVisualizer::depthCallback, this);
        // 在 DepthVisualizer 類的構造函數中，保持使用 depth_anything 的輸出
	sub_ = nh_.subscribe("/depth_anything/depth_registered/image_rect", 1, 
                    	     &DepthVisualizer::depthCallback, this);
        
        // 創建發布者
        top_view_pub_ = nh_.advertise<sensor_msgs::Image>("top_view", 1);
        colored_depth_pub_ = nh_.advertise<sensor_msgs::Image>("colored_depth", 1);

        // 初始化PCL視窗
        viewer_.setBackgroundColor(0, 0, 0);
        viewer_.addCoordinateSystem(1.0);
    }

    void depthCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        try {
            // 轉換ROS圖像到OpenCV格式
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            
            // 創建深度可視化圖像
            cv::Mat depth_colored = createDepthVisualization(cv_ptr->image);
            
            // 創建俯視圖
            cv::Mat top_view = createTopView(cv_ptr->image);

            // 顯示圖像
            cv::imshow("Depth Image", depth_colored);
            cv::imshow("Top View", top_view);
            
            // 發布圖像
            sensor_msgs::ImagePtr top_view_msg = 
                cv_bridge::CvImage(std_msgs::Header(), "bgr8", top_view).toImageMsg();
            sensor_msgs::ImagePtr colored_depth_msg = 
                cv_bridge::CvImage(std_msgs::Header(), "bgr8", depth_colored).toImageMsg();
            
            top_view_msg->header.stamp = ros::Time::now();
            colored_depth_msg->header.stamp = ros::Time::now();
            
            top_view_pub_.publish(top_view_msg);
            colored_depth_pub_.publish(colored_depth_msg);

            // 生成並顯示點雲
            createPointCloud(cv_ptr->image);

            cv::waitKey(1);

        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher top_view_pub_;
    ros::Publisher colored_depth_pub_;
    
    float depth_scale_;
    float max_depth_;
    int x_width_;
    int z_height_;
    int sample_step_;
    float fx_, fy_, cx_, cy_;
    
    cv::Mat current_depth_image_;
    pcl::visualization::PCLVisualizer viewer_;

    cv::Mat createDepthVisualization(const cv::Mat& depth_meters)
    {
        cv::Mat normalized;
        cv::normalize(depth_meters, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::Mat colored;
        cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);
        return colored;
    }

    cv::Mat createTopView(const cv::Mat& depth_meters)
    {
        cv::Mat top_view = cv::Mat::zeros(z_height_, x_width_, CV_8UC3);

        for (int y = 0; y < depth_meters.rows; y += sample_step_) {
            for (int x = 0; x < depth_meters.cols; x += sample_step_) {
                float depth = depth_meters.at<float>(y, x);
                
                if (depth > 0 && depth < max_depth_) {
                    cv::Point3f point3d = deprojectPixelToPoint(x, y, depth);
                    
                    int X = static_cast<int>(point3d.x * 100 + x_width_ / 2);
                    int Z = z_height_ - static_cast<int>(point3d.z * 100);

                    if (X >= 0 && X < x_width_ && Z >= 0 && Z < z_height_) {
                        cv::Vec3b color = getDepthColor(depth);
                        cv::circle(top_view, cv::Point(X, Z), 1, color, -1);
                    }
                }
            }
        }

        drawGrid(top_view);
        return top_view;
    }

    void drawGrid(cv::Mat& top_view)
    {
        int grid_interval = 50;
        for (int i = 0; i < x_width_; i += grid_interval) {
            cv::line(top_view, cv::Point(i, 0), cv::Point(i, z_height_-1), 
                    cv::Scalar(50,50,50), 1);
        }
        for (int i = 0; i < z_height_; i += grid_interval) {
            cv::line(top_view, cv::Point(0, i), cv::Point(x_width_-1, i), 
                    cv::Scalar(50,50,50), 1);
        }

        cv::line(top_view, 
                cv::Point(x_width_/2, 0), 
                cv::Point(x_width_/2, z_height_-1), 
                cv::Scalar(0,255,0), 2);

        for (int i = 0; i < static_cast<int>(max_depth_); ++i) {
            int y_pos = i * 100;
            if (y_pos >= 0 && y_pos < z_height_) {
                cv::putText(top_view, 
                           std::to_string(i) + "m",
                           cv::Point(x_width_-40, z_height_ - y_pos),
                           cv::FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           cv::Scalar(255,255,255), 
                           1);
            }
        }
    }

    void createPointCloud(const cv::Mat& depth_meters)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        for (int y = 0; y < depth_meters.rows; y += sample_step_) {
            for (int x = 0; x < depth_meters.cols; x += sample_step_) {
                float depth = depth_meters.at<float>(y, x);
                
                if (depth > 0 && depth < max_depth_) {
                    pcl::PointXYZRGB point;
                    point.x = (x - cx_) / fx_ * depth;
                    point.y = (y - cy_) / fy_ * depth;
                    point.z = depth;

                    cv::Vec3b color = getDepthColor(depth);
                    point.r = color[2];
                    point.g = color[1];
                    point.b = color[0];
                    
                    cloud->points.push_back(point);
                }
            }
        }

        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;

        viewer_.removeAllPointClouds();
        viewer_.addPointCloud<pcl::PointXYZRGB>(cloud, "depth cloud");
        viewer_.spinOnce();
    }

    cv::Vec3b getDepthColor(float depth)
    {
        float normalized_depth = 1.0f - std::min(depth / max_depth_, 1.0f);
        float hue = 120 + 120 * normalized_depth;
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        return rgb.at<cv::Vec3b>(0, 0);
    }

    cv::Point3f deprojectPixelToPoint(float x, float y, float depth)
    {
        float x_world = (x - cx_) / fx_ * depth;
        float y_world = (y - cy_) / fy_ * depth;
        return cv::Point3f(x_world, y_world, depth);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "depth_visualizer");
    DepthVisualizer visualizer;
    ros::spin();
    return 0;
}
