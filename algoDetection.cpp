#include <windows.h>
#include <thread>
#include <mutex>
#include <atomic>

#include <iostream>
#include <chrono>
#include  <numeric>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/videoio.hpp"
#ifndef NOCUDAPC
#include <opencv2/cudaarithm.hpp>
#endif 
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


#ifndef NOCUDAPC
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
#else
#ifdef _DEBUG
#pragma comment(lib, "opencv_world454d.lib")
#else
#pragma comment(lib, "opencv_world454.lib")
#endif 
#endif 

// GLOBALS
std::atomic <int> g_detectionState = 0;


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


void setConfigDefault(Config &params)
{
	params.debugLevel = 0;
	params.showTruck = 0;
	params.modelFolder = "C:/SRC/BauoSafeZone/config_files/";
	params.motionType = 1;
	params.MLType = 10;
	params.MHistory = 100;
	params.MvarThreshold = 580.0;
	params.MlearningRate = -1;
	params.skipMotionFrames = 1;
	params.skipDetectionFrames = 3;
	params.skipDetectionFrames2 = 3;
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
	conf.skipDetectionFrames2 = pt.get<int>("ALGO.stepDetection2", conf.skipDetectionFrames2);
	//---------
	// ALGO:
	//---------
	std::vector <int> motionROI_vec = to_array<int>(pt.get<std::string>("ALGO.motionROI", "0,0,0,0"));
	if (motionROI_vec[2] > 0) // width
		conf.motionROI = cv::Rect(motionROI_vec[0], motionROI_vec[1], motionROI_vec[2], motionROI_vec[3]);

	conf.modelFolder = pt.get<std::string>("ALGO.modelFolder", conf.modelFolder);
	conf.scale = pt.get<float>("ALGO.scale", conf.scale);
	conf.MHistory = pt.get<int>("ALGO.MHistory", conf.MHistory);
	conf.MvarThreshold = pt.get<float>("ALGO.MvarThreshold", conf.MvarThreshold);
	conf.MlearningRate = pt.get<float>("ALGO.MlearningRate", conf.MlearningRate);
	conf.motionType = pt.get<int>("ALGO.motion", conf.motionType);
	conf.trackerType = pt.get<int>("ALGO.tracker", conf.trackerType);
	conf.MLType = pt.get<int>("ALGO.ML", conf.MLType);
	conf.useGPU = pt.get<int>("ALGO.useGPU",1) ;

	
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



	void debugSaveParams(int w, int h, int imgSize, int pixelWidth, float scaleDisplay, Config params)
	{
		std::ofstream debugFile("c:\\tmp\\algoapi.txt");

		debugFile << w << " , " << h << " , " << imgSize << " , " << pixelWidth << " , " << scaleDisplay << " , " << params.useGPU << " , " << params.MLType << " , " << params.skipDetectionFrames << " , " << params.skipDetectionFrames2 << "\n";
		debugFile.close();

	}

	void ERROR_BEEP()
	{
		Beep(1200, 1200); // error beep
		Beep(1200, 1200); // error beep
	}



/*---------------------------------------------------------------------------------------------
 *						D E T E C T O R       C L A S S  
 *--------------------------------------------------------------------------------------------*/
/*
bool CDetector::InitGPU()
{

	if (!m_yolo.init(m_params.modelFolder,true)) 
	{
		std::cout << "Cant init YOLO5 net , quit \n";
		return false;
	}
	return true;
	
}
*/

bool CDetector::init(int w, int h, int imgSize , int pixelWidth, float scaleDisplay)
	{
		m_width = w;
		m_height = h;
		m_colorDepth = imgSize / (w*h);

		setConfigDefault(m_params);
		readConfigFile("C:\\Program Files\\Bauotech\\Dll\\Algo\\config.ini", m_params);
		debugSaveParams(w, h, imgSize, pixelWidth, scaleDisplay, m_params);


		if (1)
			m_colorDepth = imgSize / (w * h);
		else
			m_colorDepth = 4; //  pixelWidth / 8;  DDEBUG CONST

	

		if (m_params.useGPU == 0) 
			m_isCuda = false;
		else 
			m_isCuda = checkForGPUs() > 0;	 


#if 0		if (m_isCuda)
			MessageBoxA(0, "RUN WITH GPU", "Info", MB_OK);
		else 
			MessageBoxA(0, "RUN WITOUT GPU", "Info", MB_OK);
#endif 



		//if (m_params.MLType > 0)  {

		if (!m_yolo.init(m_params.modelFolder, m_isCuda)) {
				std::cout << "Cant init YOLO5 net , quit \n";

				return false;
		}

			m_yolo_Th = std::thread(&CYolo5::process, &m_yolo, std::ref(m_frameYolo), std::ref(m_Youtput), std::ref(g_detectionState)); // Including day cam (missing : Pass camera parameters )
		//}

		// MOG2 
		if (m_params.motionType > 0)
		{
			int emphasize = CONSTANTS::MogEmphasizeFactor;
			m_bgSeg.init(m_params.MHistory, m_params.MvarThreshold, false, emphasize);
			m_bgSeg.setLearnRate(m_params.MlearningRate);
		}	

		if (1) {
			m_concluder.init(m_params.debugLevel);			
			m_concluder.setPersonDim(MAX_PERSON_DIM); // DDEBUG CONST
		}


		// Alloce buffer
		size_t sizeTemp(m_width * m_height * m_colorDepth);
		if (m_data == NULL)
			m_data = malloc(sizeTemp);

		return true;
	}


void CDetector::terminate()
{
	g_detectionState.store(DETECTION_STATE::Terminate);
	m_yolo_Th.join();
}


	int CDetector::processFrame(cv::Mat &frame_)
	{
		// Reset process flags
		m_bgMask.setTo(0);
		m_BGSEGoutput.clear();
		m_motionDetectet = 0;


		cv::Mat frame;
		if (frame_.channels() == 4) {
			cv::cvtColor(frame_, frame, cv::COLOR_BGRA2BGR);
		}
		else
			frame = frame_;

		if (0)
		{
			imshow("DLL benny", frame);
			cv::waitKey(1);
		}


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

		// YOLO detection - send frame to yolo thread 
		// Yolo pop
		//-----------------------------------------------

		if (timeForDetection()) {
			// Yolo push 
			m_frameYolo.release();
			m_Youtput.clear();

			m_frameYolo = frame.clone();
#ifndef OFFLINE
			g_detectionState.store(DETECTION_STATE::ImageReady);
#else
			m_yolo.detect(m_frameYolo, m_Youtput);
#endif 
		}
		

		if (g_detectionState.load() == DETECTION_STATE::DetectionDone) {
			m_concluder.add(m_BGSEGoutput, m_Youtput, m_frameNum); // add & match
			; // do cunclding with yolo results.....
			g_detectionState.store(DETECTION_STATE::Idle);
		}
		else 
			m_concluder.add(m_BGSEGoutput, std::vector<YDetection>(), m_frameNum);


		m_concluder.track(); // consolidate detected objects 


		if (m_params.debugLevel > 0 &&  !m_bgMask.empty())
			cv::imshow("m_bgMask", m_bgMask);

		int tracked_count;
		tracked_count = m_BGSEGoutput.size();
		return tracked_count;
	}




	int CDetector::process(void *dataOrg, ALGO_DETECTION_OBJECT_DATA *pObjects)
	{
		
		/*
		size_t sizeTemp(m_width * m_height * m_colorDepth);
		if (m_data == NULL)
			m_data  = malloc(sizeTemp);
		*/


		//memcpy(m_data, dataOrg, sizeTemp); // buffering NOT optimized
		m_data = dataOrg; // No buffering - use original buffer for processing 

		m_frameOrg = cv::Mat(m_height, m_width, depth2cvType(m_colorDepth), m_data);
		if (m_frameOrg.empty()) {
			std::cout << "read() got an EMPTY frame\n";
			ERROR_BEEP();
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


		pObjects->reserved1_personsCount = m_concluder.getPersonObjects(m_frameNum).size();
		pObjects->reserved2_motion = objects_tracked; //  m_motionDetectet ? 1 : 0;

		// Draw overlays :
		//draw(m_frameOrg, m_Youtput , 1.0 / m_params.scale);
		if (m_params.showMotion)
			draw(m_frameOrg, m_BGSEGoutput, 1.0 / m_params.scale);
		
		draw(m_frameOrg, 1.0 / m_params.scale);

		drawInfo(m_frameOrg);
	
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

		if (0)
			cv::putText(m_frameOrg, std::to_string(m_frameNum) , cv::Point(20, img.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 1., cv::Scalar(0, 255, 0));


	}


	bool CDetector::motionDetected(cv::Mat mask)
	{ 
		return cv::countNonZero(mask) > ALGO_DETECTOPN_CONSTS::MIN_PIXELS_FOR_MOTION;
	}


	// Get motion (bgseg) interval 
	bool CDetector::timeForMotion() 
	{ 
		return (m_params.motionType > 0 && m_frameNum % m_params.skipMotionFrames == 0);
	}
	
	// Get detection (YOLO) interval 
	bool CDetector::timeForDetection()
	{
		if (m_params.MLType <= 0) 
			return false;

		if (g_detectionState.load() == DETECTION_STATE::DetectionDone)
			return false; // result was not handle (by concluder)


		if (m_concluder.numberOfPersonsObBoard() > 0)
			if (m_frameNum % m_params.skipDetectionFrames == 0)
				return true;
		if (m_BGSEGoutput.size() > 0) {
			if (m_frameNum % m_params.skipDetectionFrames == 0)
				return true;
		}
		if (m_BGSEGoutput.size() > 0) {
			if (m_frameNum %  m_params.skipDetectionFrames == 0)
				return true;
		}
		else if (m_frameNum %  m_params.detectionInterval == 0)
				return true;

		// otherwise 
		return   false;

	}

	int CDetector::getDetectionCount()
	{
		return m_concluder.getPersonObjects(m_frameNum).size(); // not optimized !!!
	}

