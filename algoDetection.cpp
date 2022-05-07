#include <thread>
#include <mutex>
#include <iostream>
#include <chrono>
#include  <numeric>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/videoio.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/lexical_cast.hpp> 

#include "yolo/yolo5.hpp"
//#include "trackerBasic.hpp"
#include "yolo/concluder.hpp"

CYolo5 _yolo;
CConcluder _concluder;


#include "utils.hpp"
#include "config.hpp"

/*
#include "mog.hpp"
#include "trackerBasic.hpp"
#include "MotionTrack.hpp"
#include "prediction.hpp"
*/

#include "algoDetection.hpp"


#ifdef _DEBUG
#pragma comment(lib, "opencv_world454d.lib")
#else
#pragma comment(lib, "opencv_world454.lib")
#endif


/*---------------------------------------------------------------------------------------------
								U T I L S
---------------------------------------------------------------------------------------------*/

const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };

std::vector <CRoi2frame>  readROISfile(std::string fname)
{
	std::vector <CRoi2frame>  rois2frames;
	//int frameNum;
	ifstream roisFile(fname);

	if (!roisFile.is_open())
		return std::vector <CRoi2frame>();

	CRoi2frame newBBox2f;
	int index;
	while (!roisFile.eof()) {
		roisFile >> index >> newBBox2f.frameNum >> newBBox2f.bbox.x >> newBBox2f.bbox.y >> newBBox2f.bbox.width >> newBBox2f.bbox.height;
		if (index > 0 && !roisFile.eof())
			rois2frames.push_back(newBBox2f);
	}

	return rois2frames;
}



bool readConfigFile(std::string ConfigFName, Config &conf)
{
	if (!FILE_UTILS::file_exists(ConfigFName))  {
		std::cout << "WARNING : Can't find Config.ini file, use default values \n";
		return false;
	}

	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini(ConfigFName, pt);
	// [GENERAL]
	conf.videoName = pt.get<std::string>("GENERAL.video", conf.videoName);
	conf.roisName = pt.get<std::string>("GENERAL.rois", conf.roisName);
	conf.waitKeyTime = pt.get<int>("GENERAL.delay-ms", conf.waitKeyTime);
	//conf.displayScale = pt.get<float>("GENERAL.scaleDisplay", conf.displayScale);
	conf.record = pt.get<int>("GENERAL.record", conf.record);
	conf.demoMode = pt.get<int>("GENERAL.demo", conf.demoMode);
	conf.debugLevel = pt.get<int>("GENERAL.debug", conf.debugLevel);
	conf.shadowclockDirection = pt.get<float>("GENERAL.shadow-hand", conf.shadowclockDirection);
	conf.showTime = pt.get<int>("GENERAL.show-time", conf.showTime);
	conf.showBoxesNum = pt.get<int>("GENERAL.show-boxes-num", conf.showBoxesNum);

	
	//---------
	// ALGO:
	//---------
	conf.modelFolder = pt.get<std::string>("ALGO.modelFolder", conf.modelFolder);
	conf.scale = pt.get<float>("ALGO.scale", conf.scale);
	conf.MHistory = pt.get<int>("ALGO.MHistory", conf.MHistory);
	conf.MvarThreshold = pt.get<float>("ALGO.MvarThreshold", conf.MvarThreshold);
	conf.MlearningRate = pt.get<float>("ALGO.MlearningRate", conf.MlearningRate);
	conf.trackerType = pt.get<int>("ALGO.tracker", conf.trackerType);
	conf.prediction = pt.get<int>("ALGO.predict", conf.prediction);
	conf.onlineTracker = pt.get<int>("ALGO.onlineTracker", conf.onlineTracker);


	return true;
}




/*---------------------------------------------------------------------------------------------
 ---------------------------------------------------------------------------------------------*/

bool CDetector::init(int w, int h, int imgSize , bool isCuda, float scaleDisplay)
	{
		m_width = w;
		m_height = h;
		m_colorDepth = imgSize / (w*h);

		readConfigFile("config.ini", m_params);

		if (!m_yolo.init(m_params.modelFolder,  isCuda)) {
			std::cout << "Cant init YOLO5 net , quit \n";
			return false;
		}

#if 0
		// MOG2 
		bool detectShadows = false; // Warning: High performance consumer 
		int emphasize = CONSTANTS::MogEmphasizeFactor;
		m_bgSeg.init(m_params.MHistory , m_params.MvarThreshold ,detectShadows, emphasize);
		m_bgSeg.setLearnRate(m_params.MlearningRate);
#endif 

		// DDEBUG DDEBUG : Read RIO's from a file 
		if (!m_params.roisName.empty())
			m_roiList = readROISfile(m_params.roisName);
		if (m_params.scale != 1.)
			for (auto &roi : m_roiList)
				roi.bbox = scaleBBox(roi.bbox, m_params.scale);

		return true;
	}


	int depth2cvType(int depth)
	{
		
		switch (depth) {
		case 1:
			return  CV_8UC1;
			break;
		case 2:
			return  CV_8UC2;
			break;
		case 3:
			return  CV_8UC3;
			break;
		case 4:
			return  CV_8UC4;
			break;
		}
	}

	int CDetector::processFrame(cv::Mat &frame)
	{
		bool is_cuda = false;

		m_Youtput.clear();
		m_yolo.detect(frame, m_Youtput);
		//_concluder.add(output);

		m_frameNum++;

		int tracked_count = 0;
		return tracked_count;
	}


	int CDetector::process(void *dataOrg)
	{
		size_t sizeTemp(m_width * m_height * m_colorDepth); 
		if (m_data == NULL)
			m_data  = malloc(sizeTemp);

		//memcpy(m_data, dataOrg, sizeTemp); // buffering NOT optimized
		m_data = dataOrg; // No buffering - use original buffer for processing 

		m_frameOrg = cv::Mat(m_height, m_width, depth2cvType(m_colorDepth), m_data);
		if (m_frameOrg.empty())
			std::cout << "read() got an EMPTY frame\n";
		else
			cv::resize(m_frameOrg, m_frame, cv::Size(0, 0), m_params.scale, m_params.scale); // performance issues 

		if (m_frameNum % m_params.detectionFPS == 0) 
			processFrame(m_frame);
		else
			int debug = 10;

		draw(m_frameOrg, 1.0 / m_params.scale);

	
		return m_frameNum++;
	}


	void CDetector::draw(cv::Mat &img, float scale)
	{


		for (int i = 0; i < m_Youtput.size(); ++i) {
			auto detection = m_Youtput[i];
			if (detection.class_id == Classes::person)
			{
				auto box = scaleBBox(detection.box, scale);
				auto classId = detection.class_id;
				const auto color = colors[classId % colors.size()];
				cv::rectangle(img, box, color, 3);

				cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
				cv::putText(img, _yolo.getClassStr(classId).c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}
	}