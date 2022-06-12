#include <windows.h>
#include <fstream>
#include <filesystem>
#include <atomic>

#include <opencv2/opencv.hpp>
#include "yolo5.hpp"



cv::Rect scaleMe(cv::Rect r, float scale, cv::Size imgSize)
{
    if (scale == 0)
        return r;

    float sWidth = (float)r.width * scale;

    float sHeight  = (float)r.height * scale;
    cv::Rect sRect = r;    
    sRect.x =  r.x - int( (sWidth - float(r.width)) * 0.5);
    sRect.y =  r.y - int( (sHeight - float(r.height)) * 0.5);

    // Check box limits:
    sRect.x =  MAX(0,sRect.x);
    sRect.y =  MAX(0,sRect.y);
    if (r.x + r.width > imgSize.width)
        r.width =  imgSize.width - r.x - 1; 
    if (r.y + r.height > imgSize.height)
        r.height =  imgSize.height - r.y - 1; 


    sRect.width = int(sWidth);
    sRect.height = int(sHeight);

    return sRect;
}
//=======================================================================

std::vector<std::string> CYolo5::load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs(m_modelFolder + "/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

bool CYolo5::load_net(bool is_cuda)
{
	std::filesystem::path cwd = std::filesystem::current_path();
		
	auto result = cv::dnn::readNet(m_modelFolder + "/yolov5s.onnx");

    if (result.empty())
        return false;

    //auto result = cv::dnn::readNet("config_files/yolov5n.onnx");
    if (is_cuda)  {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);// Note the FP16 !!!
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    m_net = result;

    return true;
}


const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;


cv::Mat CYolo5::format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}


bool CYolo5::init(std::string modelFolder, bool is_cuda)
{
    m_modelFolder = modelFolder;
    m_is_cuda = is_cuda;
    return true;
}


bool CYolo5::init()
{
	
    //m_modelFolder = modelFolder;
    m_class_list = load_class_list();
	if (m_class_list.empty()) {
		std::cout << "Error : Empty class list - ML detection won't work! \n";
		return false;
	}
    return load_net(m_is_cuda);
}



void CYolo5::process(cv::Mat &image, std::vector<YDetection> &output, std::atomic<int> &detectionState)
{
	bool isCuda = false;
	std::string modelFolder = "G:/src/BauoSafeZone/config_files/";
	//init(modelFolder, isCuda);
    init();

	while (detectionState.load() != DETECTION_STATE::Terminate) {
		if (detectionState.load() == DETECTION_STATE::ImageReady) {
			// process here.....
			//Beep(1200, 30);
			detect(image, output);
			if (detectionState.load() != DETECTION_STATE::Terminate) // atom race !!!
				detectionState.store(DETECTION_STATE::DetectionDone);
		}
	}
}


void CYolo5::detect()
{
	//detectionState.store(1);
}


/*--------------------------------------------------------------------------------------------------------------------------
        Main YDetection function:
  --------------------------------------------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------------
 *  	Relevants are only the (part of the) first 8 class IDs:
 *		person (0) ,bicycle, car,  motorbike,  aeroplane,  bus,   train,   truck (7),
 *		(not relevants: boat,traffic_light,....)
 *---------------------------------------------------------------------------------*/
const int truck_is = 7;
inline bool isClassRelevant(int class_id)
{
	return  (class_id <= truck_is);
}


void CYolo5::detect(cv::Mat &image, std::vector<YDetection> &output)
{
    cv::Mat blob;

    auto input_image = format_yolov5(image);
    
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    m_net.setInput(blob);
    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

	int dimScale = 1;

	const int dimensions = 85* dimScale; // 85;
    const int rows = 25200/ dimScale;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float * classes_scores = data + 5;
            cv::Mat scores(1, m_class_list.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD && isClassRelevant(class_id.x)) { // Worning : only first 7 classes are count!!!!

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += dimensions;

    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
		YDetection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}


/*----------------------------------------------------------------------------------------------------------------------
 * Wrraper function for process on ROI  
  ----------------------------------------------------------------------------------------------------------------------*/
void CYolo5::detect(cv::Mat &image,  std::vector<YDetection> &output_all, std::vector <cv::Rect>  ROIs)
{
    std::vector<YDetection> output;

    if (ROIs.empty()) {
        detect(image, output_all);
        return;
    }

    int maxObjects = 100;

    for (auto roi : ROIs) {
        /*
        cv::Rect sROI = scaleMe(ROIs[i], 2.0, image.size());
        cv::Mat imageROI = image(sROI); // DDEBUG [0]
        */
       cv::Mat imageROI = image(roi);

        detect(imageROI, output);

        // Return box ccordinates to full image
        for (auto &obj : output)
            obj.box += cv::Point(roi.tl());

        std::copy(output.begin(), output.end(), back_inserter(output_all));
        
        output.clear();
    }


}

