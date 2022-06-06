#pragma once
#include <atomic>


struct YDetection
{
    int class_id;
    float confidence;
    cv::Rect box;
};


enum DETECTION_STATE {
	Idle = 0,
	ImageReady, // 1
	DetectionDone, //  2
	Terminate = 9
};


class CYolo5 {
public:
	bool init(std::string modelFolder, bool is_cuda);
	void process(cv::Mat &image, std::vector<YDetection> &output, std::atomic<int> &detectionState);
	void detect(cv::Mat &image, std::vector<YDetection> &output);
	void detect(cv::Mat &image, std::vector<YDetection> &output, std::vector <cv::Rect>  ROIs);
	void detect(); // thread detection call
	std::string  getClassStr(int i) { return (m_class_list.size() > i ? m_class_list[i] : "None");}

private:
    std::vector<std::string> load_class_list();
    cv::Mat format_yolov5(const cv::Mat &source);
    bool load_net(bool is_cuda);
	bool init();

private:

    std::vector<std::string> m_class_list;
	std::string m_modelFolder;
	bool m_is_cuda = true;
    cv::dnn::Net m_net;


};

