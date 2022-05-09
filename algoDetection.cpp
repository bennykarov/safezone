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

#include "AlgoApi.h"
#include "yolo/yolo5.hpp"
//#include "trackerBasic.hpp"
#include "yolo/concluder.hpp"


#include "utils.hpp"
#include "config.hpp"

/*
#include "mog.hpp"
#include "trackerBasic.hpp"
#include "MotionTrack.hpp"
#include "prediction.hpp"
*/

#include "algoDetection.hpp"


#define EXE_MODE

#ifdef _DEBUG
#pragma comment(lib, "opencv_world454d.lib")
#else
#pragma comment(lib, "opencv_world454.lib")
#endif

namespace  ALGO_DETECTOPN_CONSTS {
	const int MIN_CONT_AREA = 20 * 10;
	const int MAX_CONT_AREA = 1000 * 1000;
	const float goodAspectRatio = 0.5;
	const float aspectRatioTolerance = 0.2;
	const int   MIN_PIXELS_FOR_MOTION = 10*10;
}

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
	//std::vector <int> v = pt.get<std::vector <int>>("GENERAL.show-boxes-num");
	conf.camROI = to_array<int>(pt.get<std::string>("GENERAL.camROI", "0,0,0,0"));
	conf.beep = pt.get<int>("GENERAL.beep", conf.beep);

	//---------
	// ALGO:
	//---------
	conf.modelFolder = pt.get<std::string>("ALGO.modelFolder", conf.modelFolder);
	conf.scale = pt.get<float>("ALGO.scale", conf.scale);
	conf.MHistory = pt.get<int>("ALGO.MHistory", conf.MHistory);
	conf.MvarThreshold = pt.get<float>("ALGO.MvarThreshold", conf.MvarThreshold);
	conf.MlearningRate = pt.get<float>("ALGO.MlearningRate", conf.MlearningRate);
	conf.trackerType = pt.get<int>("ALGO.tracker", conf.trackerType);
	conf.MLType      = pt.get<int>("ALGO.ML", conf.MLType);
	conf.prediction = pt.get<int>("ALGO.predict", conf.prediction);
	conf.onlineTracker = pt.get<int>("ALGO.onlineTracker", conf.onlineTracker);

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



	int personsDetected(std::vector<YDetection> outputs)
	{
		int counter = 0;
		for (auto detection : outputs) {
			if (detection.class_id == Classes::person)
				counter++;
		}


		return counter;
	}

/*---------------------------------------------------------------------------------------------
 *						D E T E C T O R       C L A S S  
 *--------------------------------------------------------------------------------------------*/

bool CDetector::init(int w, int h, int imgSize , bool isCuda, float scaleDisplay)
	{
		m_width = w;
		m_height = h;
		m_colorDepth = imgSize / (w*h);

		readConfigFile("config.ini", m_params);

		if (m_params.MLType > 0)
			if (!m_yolo.init(m_params.modelFolder, isCuda)) {
				std::cout << "Cant init YOLO5 net , quit \n";
				return false;
			}

		// MOG2 
		if (m_params.trackerType > 0) {
			int emphasize = CONSTANTS::MogEmphasizeFactor;
			m_bgSeg.init(m_params.MHistory, m_params.MvarThreshold, false, emphasize);
			m_bgSeg.setLearnRate(m_params.MlearningRate);
		}

		m_concluder.init();






#if 0
		// DDEBUG DDEBUG : Read RIO's from a file 
		if (!m_params.roisName.empty())
			m_roiList = readROISfile(m_params.roisName);
		if (m_params.scale != 1.)
			for (auto &roi : m_roiList)
				roi.bbox = scaleBBox(roi.bbox, m_params.scale);
#endif 

		return true;
	}


	int CDetector::processFrame(cv::Mat &frame)
	{
		bool is_cuda = false;

		// Reset process flags
		m_bgMask.setTo(0);
		m_BGSEGoutput.clear();
		m_motionDetectet = 0;
		m_doDetection = 0;
		

		if (m_params.trackerType > 0) { 
			m_bgMask = m_bgSeg.process(frame);
			if (!m_bgMask.empty())
				m_motionDetectet = cv::countNonZero(m_bgMask) > ALGO_DETECTOPN_CONSTS::MIN_PIXELS_FOR_MOTION;

			// -3- Find  motion's blobs (objects)
			if (m_motionDetectet)
				m_BGSEGoutput = detectByContours(m_bgMask);
		}

		// run YOLO or not to run YOLO, this is the question:
		if (m_params.MLType > 0 && m_frameNum % CONSTANTS::SKIP_YOLO_FRAMES == 0) {
			//if (m_params.MLType > 1)
			//m_doDetection = m_frameNum % CONSTANTS::SKIP_YOLO_FRAMES == 0 ? 1 : 0; // once awhile 
			m_doDetection = m_frameNum % m_params.MLType == 0 ? 1 : 0;			
			m_doDetection += m_BGSEGoutput.size();      // if motion object was deteted
		}

		if (m_doDetection > 0) {
			//Beep(1000, 20); // DDEBUG 
			m_Youtput.clear();
			m_yolo.detect(frame, m_Youtput);
			if (personsDetected(m_Youtput))  
				Beep(900, 10);// DDEBUG 
		}

		m_concluder.addSimple(m_Youtput);
		m_concluder.process();

		//----------------------------
		// Descide when to do what :
		//----------------------------
		if (m_frameNum % CONSTANTS::SKIP_YOLO_FRAMES)
			m_doDetection = 1;
		else
			m_doDetection = 0;


		m_frameNum++;


#ifdef EXE_MODE
		if (!m_bgMask.empty())
			cv::imshow("m_bgMask", m_bgMask);
#endif

		int tracked_count;
		tracked_count = m_BGSEGoutput.size();
		return tracked_count;
	}


	int CDetector::process(void *dataOrg, ALGO_DETECTION_OBJECT_DATA *pObjects)
	{
		size_t sizeTemp(m_width * m_height * m_colorDepth); 
		if (m_data == NULL)
			m_data  = malloc(sizeTemp);

		//memcpy(m_data, dataOrg, sizeTemp); // buffering NOT optimized
		m_data = dataOrg; // No buffering - use original buffer for processing 

		m_frameOrg = cv::Mat(m_height, m_width, depth2cvType(m_colorDepth), m_data);
		if (m_frameOrg.empty()) {
			std::cout << "read() got an EMPTY frame\n";
			return -1;
		}


		// set active ROI at first time
		if (m_camROI.width == 0) {// ROI not set yet
			if (m_params.camROI[2] > 0 && m_params.camROI[3] > 0 && m_params.camROI[0] + m_params.camROI[2] < m_frameOrg.cols && m_params.camROI[1] + m_params.camROI[3] < m_frameOrg.rows) {
				m_camROI = cv::Rect(m_params.camROI[0], m_params.camROI[1], m_params.camROI[2], m_params.camROI[3]);
			}
			else
				m_camROI = cv::Rect(0, 0, m_frameOrg.cols, m_frameOrg.rows);
		}
		
		m_frameROI = m_frameOrg(m_camROI);

		cv::resize(m_frameROI, m_frame, cv::Size(0, 0), m_params.scale, m_params.scale); // performance issues 

		int objects_tracked = 0;
		if (m_frameNum % m_params.detectionFPS == 0) 
			objects_tracked = processFrame(m_frame);

		pObjects->reserved1_personsCount = 0;
		pObjects->reserved2_motion = objects_tracked; //  m_motionDetectet ? 1 : 0;

		if (objects_tracked > 0)
			int debug = 10;


		// Draw overlays :
		draw(m_frameOrg, m_Youtput , 1.0 / m_params.scale);
		draw(m_frameOrg, m_BGSEGoutput, 1.0 / m_params.scale);

		drawInfo(m_frameOrg);
		//cv::putText(m_frameOrg, std::to_string(m_frameNum) , cv::Point(200, m_frameOrg.rows - 200), cv::FONT_HERSHEY_SIMPLEX, 1., cv::Scalar(0, 255, 0));


	
		return m_frameNum++;
	}




	std::vector <cv::Rect> CDetector::detectByContours(cv::Mat bgMask)
	{
		// Find blobs using CONTOURS 
		std::vector < std::vector<cv::Point>> contours, good_contours;
		//std::vector <cv::Rect> eyeROIs;
		std::vector<cv::Rect>    newROIs;
		std::vector<cv::Vec4i> hierarchy;

		findContours(bgMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		// Contour analysis

		//m_objects.clear(); // DDEBUG 

		for (auto cont : contours) {
			cv::Rect box = cv::boundingRect(cont);
			int area = contourArea(cont, false);

			// Filters:
			if (area < ALGO_DETECTOPN_CONSTS::MIN_CONT_AREA || area > ALGO_DETECTOPN_CONSTS::MAX_CONT_AREA)
				continue;

			// FIlter outliers 
			//rbox = cv::minAreaRect(cont);
			/*
			attrib.aspectRatio = attrib.rbox.size.height / attrib.rbox.size.width;
			attrib.perimeter = cv::arcLength(cv::Mat(contours[i]), true);
			attrib.thickness = attrib.perimeter / attrib.area;
			attrib.close = (hierarchy[i][2] < 0 && hierarchy[i][3] < 0) ? -1 : 1;
			attrib.len = MAX(attrib.rbox.size.width, attrib.rbox.size.height);
			attrib.topLevel = hierarchy[i][3] == -1; // no paretn
			*/
			//cv::Rect debug = scaleBBox(box, 1. / 0.5);

			UTILS::checkBounderies(box, bgMask.size());
			newROIs.push_back(box);
		}

		return newROIs;
	}



	void CDetector::draw(cv::Mat &img, std::vector<YDetection> Youtput, float scale)
	{
		for (int i = 0; i < Youtput.size(); ++i) {
			auto detection = Youtput[i];
			if (detection.class_id == Classes::person || detection.class_id == Classes::car || 
				detection.class_id == Classes::truck || detection.class_id == Classes::bus)
			{
				auto box = scaleBBox(detection.box, scale);
				box += cv::Point(m_camROI.x, m_camROI.y);
				auto classId = detection.class_id;
				const auto color = colors[classId % colors.size()];
				cv::rectangle(img, box, color, 3);

				cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
				cv::putText(img, m_yolo.getClassStr(classId).c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}
	}

	void CDetector::draw(cv::Mat &img, std::vector<cv::Rect>  rois, float scale)
	{
		cv::Scalar color(0, 255, 0);

		for (auto roi : rois) {
			cv::Rect fixROI = scaleBBox(roi, scale) + cv::Point(m_camROI.x, m_camROI.y);
			cv::rectangle(img, fixROI, color, 2);
		}

	}

	void CDetector::drawInfo(cv::Mat &img)
	{
		cv::Scalar color(255, 0, 0);
		cv::rectangle(img, m_camROI, color, 2);

		cv::putText(m_frameOrg, std::to_string(m_frameNum) , cv::Point(20, img.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 1., cv::Scalar(0, 255, 0));


	}


	bool CDetector::motionDetected(cv::Mat mask)
	{ 
		return cv::countNonZero(mask) > ALGO_DETECTOPN_CONSTS::MIN_PIXELS_FOR_MOTION;
	}
