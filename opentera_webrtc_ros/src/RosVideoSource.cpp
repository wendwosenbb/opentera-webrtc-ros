#include "opentera_webrtc_ros/utils.h"
#include <opentera_webrtc_ros/RosVideoSource.h>
#include <api/video/i420_buffer.h>

// We use OpenCV for image buffer manipulation
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace opentera;

/**
 * @brief Construct a RosVideoSource
 *
 * @param needsDenoising denoising should be applied to the video stream by the image transport layer
 * @param isScreenCast the transport layer should be configured to stream a screen rather then a camera
 */
RosVideoSource::RosVideoSource(bool needsDenoising, bool isScreenCast)
    : VideoSource(VideoSourceConfiguration::create(needsDenoising, isScreenCast))
{
    nvjpegCreateSimple(&nvjpeg_handle_);
    nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_);
}

RosVideoSource::~RosVideoSource()
{
    nvjpegJpegStateDestroy(nvjpeg_state_);
    nvjpegDestroy(nvjpeg_handle_);
}
/**
 * @brief Process a frame received from ROS
 *
 * We grabbed the code from here and added even frame check
 * https://github.com/RobotWebTools/webrtc_ros/blob/develop/webrtc_ros/src/ros_video_capturer.cpp#L48
 *
 * @param msg The ROS image message
 */
void RosVideoSource::sendFrame(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    cv::Mat bgr;
    if (msg->encoding.find("F") != std::string::npos)
    {
        // scale floating point images
        cv::Mat float_image_bridge = cv_bridge::toCvShare(msg, msg->encoding)->image;
        cv::Mat_<float> float_image = float_image_bridge;
        double max_val;
        cv::minMaxIdx(float_image, 0, &max_val);

        if (max_val > 0)
        {
            float_image *= (255 / max_val);
        }
        cv::Mat orig;
        float_image.convertTo(orig, CV_8U);
        cv::cvtColor(orig, bgr, CV_GRAY2BGR);
    }
    else
    {
        bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    }

    int64_t camera_time_us = to_microseconds(msg->header.stamp);
    VideoSource::sendFrame(bgr, camera_time_us);
}

// void RosVideoSource::sendcompressedFrame(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg)
// {
//     // Decode the compressed image to a cv::Mat
//     cv::Mat compressed_image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    
//     if (compressed_image.empty())
//     {
//         RCLCPP_ERROR(rclcpp::get_logger("RosVideoSource"), "Failed to decode compressed image");
//         return;
//     }

//     // Convert to BGR format if needed
//     cv::Mat bgr;
//     if (msg->format.find("mono") != std::string::npos)  // Check if it's grayscale
//     {
//         cv::cvtColor(compressed_image, bgr, cv::COLOR_GRAY2BGR);
//     }
//     else
//     {
//         bgr = compressed_image;
//     }

//     int64_t camera_time_us = to_microseconds(msg->header.stamp);
//     VideoSource::sendFrame(bgr, camera_time_us);
// }

void RosVideoSource::sendcompressedFrame(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg)
{
    const unsigned char* jpeg_data = msg->data.data();
    size_t jpeg_size = msg->data.size();

    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];

    nvjpegStatus_t info_status = nvjpegGetImageInfo(nvjpeg_handle_, jpeg_data, jpeg_size,
                                                    &nComponents, &subsampling, widths, heights);
    if (info_status != NVJPEG_STATUS_SUCCESS) {
        RCLCPP_ERROR(rclcpp::get_logger("RosVideoSource"), "Failed to get JPEG info");
        return;
    }

    int width = widths[0];
    int height = heights[0];

    // Allocate output on GPU
    uchar3* d_output;
    cudaMalloc(&d_output, width * height * sizeof(uchar3));

    nvjpegImage_t output_image;
    memset(&output_image, 0, sizeof(output_image));
    output_image.channel[0] = reinterpret_cast<unsigned char*>(d_output);
    output_image.pitch[0] = width * 3;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    nvjpegStatus_t decode_status = nvjpegDecode(nvjpeg_handle_, nvjpeg_state_,
                                                jpeg_data, jpeg_size,
                                                NVJPEG_OUTPUT_RGBI, &output_image, stream);

    if (decode_status != NVJPEG_STATUS_SUCCESS) {
        RCLCPP_ERROR(rclcpp::get_logger("RosVideoSource"), "nvJPEG decode failed");
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        return;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Wrap it in cv::cuda::GpuMat and send
    cv::cuda::GpuMat gpu_mat(height, width, CV_8UC3, d_output);
    int64_t camera_time_us = to_microseconds(msg->header.stamp);
    VideoSource::sendFrame(gpu_mat, camera_time_us);

    // Cleanup after send
    cudaFree(d_output);
}