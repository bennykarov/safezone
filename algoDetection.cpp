#include <windows.h>
#include <thread>
#include <mutex>
#include <iostream>
#include <chrono>
#include  <numeric>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/cudaarithm.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/lexical_cast.hpp> 


#include "AlgoApi.h"
#include "CObject.hpp"
#include "yolo/yolo5.hpp"
//#include "trackerBasic.hpp"
#include "concluder.hpp"


#include "utils.hpp"
#include "config.hpp"

/*
#include "mog.hpp"
#include "trackerBasic.hpp"
#include "MotionTrack.hpp"
#include "prediction.hpp"
*/

#include "algoDetection.hpp"

#define MAX_PERSON_DIM	cv::Size(40, 90) // DDEBUG CONST


#define EXE_MODE

#ifdef _DEBUG
#pragma comment(lib, "opencv_core454d.lib")
#pragma comment(lib, "opencv_highgui454d.lib")
#pragma comment(lib, "opencv_video454d.lib")
#pragma comment(lib, "opencv_videoio454d.lib")
#pragma comment(lib, "opencv_imgcodecs454d.lib")
#pragma comment(lib, "opencv_imgproc454d.lib")
#pragma comment(lib, "opencv_tracking454d.lib")
#pragma comment(lib, "opencv_dnn454d.lib")
//#pragma comment(lib, "opencv_calib3d454d.lib")
//#pragma comment(lib, "opencv_bgsegm454d.lib")
#else
#pragma comment(lib, "opencv_core454.lib")
#pragma comment(lib, "opencv_highgui454.lib")
#pragma comment(lib, "opencv_video454.lib")
#pragma comment(lib, "opencv_videoio454.lib")
#pragma comment(lib, "opencv_imgcodecs454.lib")
#pragma comment(lib, "opencv_imgproc454.lib")
#pragma comment(lib, "opencv_tracking454.lib")
#pragma comment(lib, "opencv_dnn454.lib")
//#pragma comment(lib, "opencv_world454.lib")
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
	conf.record = pt.get<int>("GENERAL.record", conf.record);
	conf.demoMode = pt.get<int>("GENERAL.demo", conf.demoMode);
	conf.debugLevel = pt.get<int>("GENERAL.debug", conf.debugLevel);
	conf.showTruck = pt.get<int>("GENERAL.showTruck", conf.showTruck);
	conf.showMotion = pt.get<int>("GENERAL.showMotion", conf.showMotion);
	conf.camROI = to_array<int>(pt.get<std::string>("GENERAL.camROI", "0,0,0,0"));
	// [OPTIMIZE]  Optimization
	conf.skipMotionFrames = pt.get<int>("ALGO.stepMotion", conf.skipMotionFrames);
	conf.skipDetectionFrames = pt.get<int>("ALGO.stepDetection", conf.skipDetectionFrames);
	//---------
	// ALGO:
	//---------
	conf.modelFolder = pt.get<std::string>("ALGO.modelFolder", conf.modelFolder);
	conf.scale = pt.get<float>("ALGO.scale", conf.scale);
	conf.MHistory = pt.get<int>("ALGO.MHistory", conf.MHistory);
	conf.MvarThreshold = pt.get<float>("ALGO.MvarThreshold", conf.MvarThreshold);
	conf.MlearningRate = pt.get<float>("ALGO.MlearningRate", conf.MlearningRate);
	conf.motionType = pt.get<int>("ALGO.motion", conf.motionType);
	conf.trackerType = pt.get<int>("ALGO.tracker", conf.trackerType);
	conf.MLType = pt.get<int>("ALGO.ML", conf.MLType);
	conf.useGPU = pt.get<int>("ALGO.useGPU", conf.useGPU > 0 ? 1:0) > 1;

	
	/*
	conf.prediction = pt.get<int>("ALGO.predict", conf.prediction);
	conf.onlineTracker = pt.get<int>("ALGO.onlineTracker", conf.onlineTracker);
	//conf.displayScale = pt.get<float>("GENERAL.scaleDisplay", conf.displayScale);
	//conf.beep = pt.get<int>("GENERAL.beep", conf.beep);
	//conf.shadowclockDirection = pt.get<float>("GENERAL.shadow-hand", conf.shadowclockDirection);
	//conf.showTime = pt.get<int>("GENERAL.show-time", conf.showTime);
	//conf.showBoxesNum = pt.get<int>("GENERAL.show-boxes-num", conf.showBoxesNum);
	//std::vector <int> v = pt.get<std::vector <int>>("GENERAL.show-boxes-num");
	*/
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
			if (detection.class_id == Labels::person)
				counter++;
		}


		return counter;
	}

	int checkForGPUs()
	{
		using namespace cv::cuda;

		std::cout << "--------------------------";
		std::cout << "GPU INFO : ";
		printShortCudaDeviceInfo(getDevice());
		int cuda_devices_number = getCudaEnabledDeviceCount();
		cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;
		
		DeviceInfo _deviceInfo;
		bool _isd_evice_compatible = _deviceInfo.isCompatible();
		cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
		std::cout << "--------------------------";
		return cuda_devices_number;
	}


/*---------------------------------------------------------------------------------------------
 *						D E T E C T O R       C L A S S  
 *--------------------------------------------------------------------------------------------*/
bool CDetector::init(int w, int h, int imgSize , float scaleDisplay)
	{
		m_width = w;
		m_height = h;
		m_colorDepth = imgSize / (w*h);


		if (!m_params.useGPU) 
			m_isCuda = false;
		else 
			m_isCuda = checkForGPUs() > 0;


		readConfigFile("config.ini", m_params);

		if (m_params.MLType > 0)
			if (!m_yolo.init(m_params.modelFolder, m_isCuda)) {
				std::cout << "Cant init YOLO5 net , quit \n";
				return false;
			}

		// MOG2 
		if (m_params.motionType > 0) {
			int emphasize = CONSTANTS::MogEmphasizeFactor;
			m_bgSeg.init(m_params.MHistory, m_params.MvarThreshold, false, emphasize);
			m_bgSeg.setLearnRate(m_params.MlearningRate);
		}

		if (1) {
			m_concluder.init(m_params.debugLevel);
			m_concluder.setPersonDim(MAX_PERSON_DIM); // DDEBUG CONST
		}
		// TESTS:
		// m_tracker.track_main("G:/data/bauoTech/Alenbi/04-04-2022/B_ch08.mp4", m_params.trackerType, 13000);   exit(0);

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
		// Reset process flags
		m_bgMask.setTo(0);
		m_BGSEGoutput.clear();
		m_motionDetectet = 0;

		// BG seg detection
		//--------------------
		if (timeForMotion()) {
			m_bgMask = m_bgSeg.process(frame);
			if (!m_bgMask.empty()) {
				m_motionDetectet = cv::countNonZero(m_bgMask) > ALGO_DETECTOPN_CONSTS::MIN_PIXELS_FOR_MOTION;
				std::vector <cv::Rect>  BGSEGoutput = detectByContours(m_bgMask);
				for (auto obj : BGSEGoutput) {
					if (obj.area() <= MAX_PERSON_DIM.width*MAX_PERSON_DIM.height)
						m_BGSEGoutput.push_back(obj);
					else 
						m_BGSEGoutputLarge.push_back(obj);
				}

			}
		}
		else
			int debug = 10;

		// YOLO detection
		//--------------------
		if (timeForDetection()) {
			//Beep(1000, 20); // DDEBUG 
			m_Youtput.clear();
			//int64 timer = getTickCount();

			/*
			if (1) {// DEBUG
				cv::Mat testImg = frame(cv::Rect(0, 0, frame.cols / 6, frame.rows / 6));
				auto tp1 = chrono::system_clock::now();
				m_yolo.detect(testImg, m_Youtput);
				auto tp2 = chrono::system_clock::now();
				chrono::duration<long double> delta__time = tp2 - tp1;
				std::cout << "Partial \ Full Yolo duration = (" << delta__time.count() << " : ";
			}
			*/

			m_yolo.detect(frame, m_Youtput);
			if (personsDetected(m_Youtput))   
				Beep(900, 10);// DDEBUG 
		}

		m_concluder.add(m_BGSEGoutput, m_Youtput, m_frameNum); // add & match

		m_concluder.track(); // consolidate detected objects 


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


		// Set active ROI at first time
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

		objects_tracked = processFrame(m_frame);

		//pObjects->reserved1_personsCount = ;
		pObjects->reserved2_motion = objects_tracked; //  m_motionDetectet ? 1 : 0;

		// Draw overlays :
		//draw(m_frameOrg, m_Youtput , 1.0 / m_params.scale);
		if (m_params.showMotion)
			draw(m_frameOrg, m_BGSEGoutput, 1.0 / m_params.scale);
		draw(m_frameOrg, 1.0 / m_params.scale);

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


	/*---------------------------------------------------------------------------------------------
	 *			DRAW FUNCTIONS 
	 ---------------------------------------------------------------------------------------------*/

	void CDetector::draw(cv::Mat &img, float scale)
	{
		cv::Scalar  color(0, 0, 0);

		for (auto obj : m_concluder.getPersonObjects(m_frameNum)) {
			auto box = scaleBBox(obj.m_bbox, scale);
			box += cv::Point(m_camROI.x, m_camROI.y);
			auto classId = obj.m_finalLabel;
			//--------------------------------------------------------------
			// Colors: 
			//        RED for person, 
			//        BLUE for moving YOLO object (other than Person), 
			//		  White for motion tracking 
			//--------------------------------------------------------------
			// Person objects
			if (obj.m_finalLabel == Labels::person)
				color = cv::Scalar(0, 0, 255); // colors[classId % colors.size()];
			// BGSeg stable objects
			else if (obj.m_finalLabel == Labels::nonLabled) // motion
				color = cv::Scalar(200, 200, 200);
			else
				continue;

			cv::rectangle(img, box, color, 3);
			// Add lablel
			if (obj.m_finalLabel != Labels::nonLabled) {
				cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
				cv::putText(img, m_yolo.getClassStr(classId).c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}


		// Draw other Labeled objects (only moving objects)
		if (m_params.showTruck > 0) {
			bool showOnlyWhileMoving = m_params.showTruck == 1;
			for (auto obj : m_concluder.getVehicleObjects(m_frameNum, showOnlyWhileMoving)) {
				auto box = scaleBBox(obj.m_bbox, scale);
				box += cv::Point(m_camROI.x, m_camROI.y);
				auto classId = obj.m_finalLabel;

				if (classId == Labels::train || classId == Labels::bus) // DDEBUg
					classId = Labels::truck;

				// Draw vehicles and othe 
				if (classId == Labels::truck) {
					color = cv::Scalar(155, 155, 0);
					cv::Point debug = centerOf(box);
					cv::putText(img, "T", centerOf(box), cv::FONT_HERSHEY_SIMPLEX, 2.5, color, 10);

				}
				else {
					color = cv::Scalar(255, 0, 0);
					cv::rectangle(img, box, color, 3);
					// Add lablel
					cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
					cv::putText(img, m_yolo.getClassStr(classId).c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
				}
			}
		}
	}

	void CDetector::draw(cv::Mat &img, std::vector<YDetection> Youtput, float scale)
	{
		for (int i = 0; i < Youtput.size(); ++i) {
			auto detection = Youtput[i];
			if (detection.class_id == Labels::person || detection.class_id == Labels::car || 
				detection.class_id == Labels::truck || detection.class_id == Labels::bus)
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


	bool CDetector::timeForMotion() 
	{ 
		return (m_params.motionType > 0 && m_frameNum % m_params.skipMotionFrames == 0);
	}
	
	bool CDetector::timeForDetection()
	{
		if (m_params.MLType <= 0) 
			return false;
		if (m_BGSEGoutput.size() > 0) {
			if (m_frameNum %  m_params.skipDetectionFrames == 0)
				return true;
		}
		else if (m_frameNum %  m_params.detectionInterval == 0)
				return true;

		// otherwise 
		return   false;

	}
