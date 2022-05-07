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

#include "utils.hpp"
#include "config.hpp"

/*
#include "mog.hpp"
#include "trackerBasic.hpp"
#include "MotionTrack.hpp"
#include "prediction.hpp"
*/

#include "algoTracker.hpp"


#ifdef _DEBUG
#pragma comment(lib, "opencv_world454d.lib")
/*
#pragma comment(lib, "opencv_core430d.lib")
#pragma comment(lib, "opencv_highgui430d.lib")
#pragma comment(lib, "opencv_imgproc430d.lib")
#pragma comment(lib, "opencv_tracking430d.lib")
#pragma comment(lib, "opencv_videoio430d.lib")
#pragma comment(lib, "opencv_video430d.lib")
#pragma comment(lib, "opencv_features2d430d.lib")
#pragma comment(lib, "opencv_videoio430d.lib")
#pragma comment(lib, "opencv_imgcodecs430d.lib")
#pragma comment(lib, "opencv_bgsegm430d.lib")
*/
#else
#pragma comment(lib, "opencv_world454.lib")
/*
#pragma comment(lib, "opencv_core430.lib")
#pragma comment(lib, "opencv_highgui430.lib")
#pragma comment(lib, "opencv_imgproc430.lib")
#pragma comment(lib, "opencv_tracking430.lib")
#pragma comment(lib, "opencv_videoio430.lib")
#pragma comment(lib, "opencv_video430.lib")
#pragma comment(lib, "opencv_features2d430.lib")
*/
#endif

int test_KF_mouse();
int debugCheckOverlapping(std::vector<CObject>   &objects);
int debugCheckOverlapping(std::vector<CObject>   &objects, cv::Rect2f roi, float threshold);



/*---------------------------------------------------------------------------------------------
								U T I L S
---------------------------------------------------------------------------------------------*/

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
	conf.scale = pt.get<float>("ALGO.scale", conf.scale);
	conf.record = pt.get<int>("GENERAL.record", conf.record);
	conf.demoMode = pt.get<int>("GENERAL.demo", conf.demoMode);
	conf.debugLevel = pt.get<int>("GENERAL.debug", conf.debugLevel);
	conf.shadowclockDirection = pt.get<float>("GENERAL.shadow-hand", conf.shadowclockDirection);
	conf.showTime = pt.get<int>("GENERAL.show-time", conf.showTime);
	conf.showBoxesNum = pt.get<int>("GENERAL.show-boxes-num", conf.showBoxesNum);
	
	//---------
	// ALGO:
	//---------
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

void CTrack::init(int w, int h, int imgSize , float scaleDisplay)
	{
		m_width = w;
		m_height = h;
		m_colorDepth = imgSize / (w*h);

		readConfigFile("config.ini", m_params);

		// MOG2 
		bool detectShadows = false; // Warning: High performance consumer 
		int emphasize = CONSTANTS::MogEmphasizeFactor;
		m_bgSeg.init(m_params.MHistory , m_params.MvarThreshold ,detectShadows, emphasize);
		m_bgSeg.setLearnRate(m_params.MlearningRate);

		// DDEBUG DDEBUG : Read RIO's from a file 
		if (!m_params.roisName.empty())
			m_roiList = readROISfile(m_params.roisName);
		if (m_params.scale != 1.)
			for (auto &roi : m_roiList)
				roi.bbox = scaleBBox(roi.bbox, m_params.scale);
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

	int CTrack::processFrame(cv::Mat &frame)
	{
		bool is_cuda = true;


		int tracked_count = 0;
		return tracked_count;
	}


	int CTrack::process(void *dataOrg)
	{
		size_t sizeTemp(m_width * m_height * m_colorDepth); 
		if (m_data == NULL)
			m_data  = malloc(sizeTemp);


		//--------------------------- T E S T     Z O N E  ------------------------------------------		
		//testKalman2();
		//test_KF_mouse();
		//-------------------------------------------------------------------------------------------

		//memcpy(m_data, dataOrg, sizeTemp); // buffering NOT optimized
		m_data = dataOrg; // No buffering - use original buffer for processing 

		m_frameOrg = cv::Mat(m_height, m_width, depth2cvType(m_colorDepth), m_data);
		if (m_frameOrg.empty())
			std::cout << "read() got an EMPTY frame\n";
		else
			cv::resize(m_frameOrg, m_frame, cv::Size(0, 0), m_params.scale, m_params.scale);

		if (m_frameNum % m_params.detectionFPS == 0) 
			processFrame(m_frame);
		else
			int debug = 10;

	
		return m_frameNum++;
	}


/*-----------------------------------------------------------------------------------------------------
 *-----------------------------------------------------------------------------------------------------
 *							(OLD) TRACKER ZONE 
 *-----------------------------------------------------------------------------------------------------
 -----------------------------------------------------------------------------------------------------*/
#if 0
	cv::Scalar  setColorByDetection(DET_TYPE type_)
	{
		int type = (int)type_;

		switch (type) {
		case (int)DET_TYPE::BGSeg:
			return cv::Scalar(0, 255, 0);
		case (int)DET_TYPE::OpticalFlow:
			return cv::Scalar(0, 0, 255);
		case (int)DET_TYPE::Prediction:
			return cv::Scalar(255, 0, 0);
		case (int)DET_TYPE::Hidden:
				return cv::Scalar(0, 0, 0);
			default:
				return cv::Scalar(255, 255, 255);
		}
	}


	cv::Scalar  setColorByLabel(LABEL l)
	{
		
		if (l == LABEL::HUMAN)
			return cv::Scalar(0, 255, 0);
		else if (l == LABEL::VEHICLE)
			return cv::Scalar(255, 0, 0);
		else
			return cv::Scalar(0, 0, 0);
	}

	cv::Scalar  setColorByDimensions(cv::Rect roi)
	{
		if (roi.width < int((float)SIZES::minHumanWidth * 0.5) ||
			roi.height < int((float)SIZES::minHumanHeight * 0.5))
			return cv::Scalar(0, 0, 0);
		else if (roi.width <= int((float)SIZES::maxHumanWidth * 0.5) &&
			roi.height <= int((float)SIZES::maxHumanHeight * 0.5) &&
			(float)roi.width / (float)roi.height < 0.8)
			return cv::Scalar(0, 255, 0);
		else // if (roi.width < int((float)SIZES::minVehicleWidth * m_params.scale))
			return cv::Scalar(255, 0, 0);

	}


	std::vector<cv::KeyPoint> CTrack::detectBySimpleBlob(cv::Mat bgMask)
	{
		std::vector<cv::KeyPoint> myBlobs;
		// Find blobs using SIMPLEBOBS
		cv::SimpleBlobDetector::Params params;
		params.minDistBetweenBlobs = 10.0;  // minimum 10 pixels between blobs

		params.filterByCircularity = false;
		params.filterByColor = false;
		params.filterByConvexity = false;
		params.filterByInertia = false;

		params.filterByArea = true;         // filter my blobs by area of blob
		params.minArea = 2 * 2;
		params.maxArea = 100 * 100;

		cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

		detector->detect(bgMask, myBlobs);

		return myBlobs;
	}



	std::vector <cv::Rect> CTrack::detectByContours(cv::Mat bgMask)
	{
		// Find blobs using CONTOURS 
		std::vector < std::vector<cv::Point>> contours, good_contours;
		//std::vector <cv::Rect> eyeROIs;
		std::vector<cv::Rect>    newROIs;
		std::vector<cv::Vec4i> hierarchy;

		findContours(bgMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		const int MIN_CONT_AREA = 20*10;
		const int MAX_CONT_AREA = 200 * 120;

		// Contour analysis
		float min_area = MIN_CONT_AREA;
		//float max_area = MAX_ROI_AREA; 
		float max_area = MAX_CONT_AREA;
		float goodAspectRatio = 0.5;
		float aspectRatioTolerance = 0.2;

		//m_objects.clear(); // DDEBUG 

		for (auto cont : contours) {
			int area = contourArea(cont, false);
			if (area < MIN_CONT_AREA || area > MAX_CONT_AREA)
				continue;
			cv::Rect box = cv::boundingRect(cont);
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



	/*-----------------------------------------------------------
	 * Match RECTs to exiting m_objects
	 * If newROI matched update BBOX
	 * unless this bbox already updated (by tracker)
	 * Return matches vecotor (0 or 1 for each newROI)
	 *-----------------------------------------------------------*/
	std::vector <int> CTrack::matchObjects(std::vector<cv::Rect> newROIs)
	{
		const float MinOverLappedRatio = 0.5;
		const float GoodOverLappedRatio = 0.7;
		const float GoodSizeRatio = 0.7;
		/*-------------------------------------------
		 * match new objects to the new ones
		 * Check box overlapping to match new & old
		 -------------------------------------------*/

		std::vector <int> match(newROIs.size(), 0);
		//match.assign(newROIs.size(), 0);
		int objInd = 0;
		for (auto &oldObj : m_objects) {
			if (oldObj.m_lastDetected == m_frameNum)
				continue;  // already detected by other algo (tracker etc.)

			cv::Rect predictBox;
			cv::Rect2f newBox;
			DET_TYPE detectionType = DET_TYPE::DETECTION_NA;
			float bestOverLapped = 0;

			if (oldObj.m_ID == 9)
				int debug = 10;

			//---------------------------------
			// Case 1 (best) - rect overlapped 
			//---------------------------------
			for (int i = 0; i < newROIs.size(); i++) {
					if (bboxesBounding(oldObj.m_bboxes.back(), newROIs[i]) > 0.3) {
					//if ((newROIs[i] & cv::Rect(oldObj.m_bboxes.back())).area() > 0) {
					// Special treat to young object (dont use prediction), only matched by boxes Similarity
					//if (bboxRatio(newROIs[i], oldObj.m_bboxes.back()) > 0.7) { // two boxes over-lappeded and with similar size 
					// Allow object to decreases area more than increases :
					if (bboxOrderRatio(oldObj.m_bboxes.back(), newROIs[i]) > 0.5 && bboxOrderRatio(oldObj.m_bboxes.back(), newROIs[i]) < 1.9) {
						newBox = newROIs[i]; // match by box similarity 
						detectionType = DET_TYPE::BGSeg;
						if (isStableDetection(oldObj, 4)) // reset current prediction obj (if exist) for next time
							m_predictions[objInd].resetKF2();
						match[i]++; // Allow multiple matching 
						break; // out of ROI loop 
					}
				}
			}

					//------------------------------------------
					// Case 2:  Matched by Tracker (Optical)
					//------------------------------------------
/*
					else if (m_params.onlineTracker > 0 && oldObj.m_detectionStatus >= STAGE::STABLE &&
						!UTILS::nearEdges(m_frame.size(), oldObj.m_bboxes.back())) {						
						static CMotionTracker trackerOptical;
						cv::Rect trackBox = trackerOptical.track(m_frame, m_prevFrame, resizeBBox(oldObj.m_bboxes.back(), m_frame.size(), 2.0));
						if (!trackBox.empty()) {
							newBox = moveByCenter(oldObj.m_bboxes.back(), centerOf(trackBox)); // use trtacker box just for its center 
							detectionType = DET_TYPE::OpticalFlow;
							match[i]++; // Allow multiple matching 
							break; // out of ROI loop 
						}
					}
*/ 
				//------------------------------------------------------------------
				// Case 3  - poor overlapping  (& no tracking) - use prediction 
				//------------------------------------------------------------------
			if (detectionType == DET_TYPE::DETECTION_NA && m_params.prediction > 0 && 
					oldObj.len() > CONSTANTS::StableLenForPrediction)  {
				for (int i = 0; i < newROIs.size(); i++) {
					if ((newROIs[i] & cv::Rect(oldObj.m_bboxes.back())).area() > 0) { // Here : overlapping is required even for prediction 
						float confidence;
						// Kalman 1  predictBox = m_predictions[objInd].predictKF(oldObj, confidence);
						cv::Point2f predictCenter = m_predictions[objInd].predictKF2(m_objects[objInd] /*oldObj*/, confidence, m_frameNum);
						if (predictCenter.x >= 0 && predictCenter.y >= 0) {
							predictBox = moveByCenter(/*oldObj*/m_objects[objInd].m_bboxes.back(), predictCenter);
							float curOverLapped = bboxesBounding(predictBox, newROIs[i]); // predictBox bounding-ratio inside newROI
							//if (1) {//if (curOverLapped > MinOverLappedRatio)   DDEBUG REMOVED CONDITION 
							newBox = predictBox;
							//newBox = predictBox & newROIs[i];
							detectionType = DET_TYPE::Prediction;
							match[i]++; // Allow multiple matching 
							break; // out of ROI loop 
						}
						//else  std::cout << "Predicted object got lost here ! \n";
					}
				}
			}// ROI loop 

		if (detectionType > DET_TYPE::DETECTION_NA) {
			blenBbox(newBox, oldObj.m_bboxes.back(), 0.2);
			oldObj.add(newBox, m_frameNum, detectionType);
		}
		else
			std::cout << "we lost an object \n";

		objInd++;
		}
	
		return match;
	}




	/*------------------------------------------------------------------------------------------
	 * Set misc parameters: detection score, ..
	 ------------------------------------------------------------------------------------------*/
	void CTrack::consolidateDetection()
	{
		const int MaxHiddenTrackFrames = 5;
		const int starterLen = 10;   // DDEBUG CONST 
		const int stableLen = CONSTANTS::FPS * 1; // DDEBUG CONST 

		// Set stability score
		for (auto &obj : m_objects) {
			obj.m_detectionStatus = STAGE::STARTER;
			if (obj.len() > stableLen) {
				if (isStable(obj, CONSTANTS::StableLen))
					obj.m_detectionStatus = STAGE::STABLE;
				else 
					obj.m_detectionStatus = STAGE::SENIOR;

			}
			else if (obj.len() > starterLen)
				obj.m_detectionStatus = STAGE::FINE;
		}
		// Remove un-detected objects (fade)
		for (int i = 0; i < m_objects.size(); i++) {
			if (m_objects[i].m_bboxes.size() > 10)
				int debug = 10;
			int hiddenLenAllowed = m_objects[i].m_bboxes.size() < 4 ? 3 : 10;
			if (m_objects[i].m_lastDetected + hiddenLenAllowed < m_frameNum)
				m_objects.erase(m_objects.begin() + i--);
		}

		// re Classify objects 
		for (auto obj : m_objects)
			obj.m_label = classify(m_frame, m_bgMask, obj.m_bboxes.back());

	}

#endif
#if 0
	/*-----------------------------------------------------------------------
		Check stability of detection while in last LEN frames:
		(1) Not (even once) use predicted position
		(2) Diff for consequences areas are less than 0.3 (CONST!)
	 -----------------------------------------------------------------------*/
	bool CTrack::isStable(const CObject &obj, int len)
	{

		if (obj.m_bboxes.size() < len)
			return false;

		int ind = obj.m_bboxes.size() - 1;
		for (int i = 0; i < len; i++, ind--)
			if (obj.m_detectionTypes[i] <= DET_TYPE::Prediction)
			return false;

		// Check for rect stability 
		int areaDiff = 0;
		ind = obj.m_bboxes.size() - 1;
		for (int i = 0; i < len - 1; i++,ind--)
			areaDiff += obj.m_bboxes[ind].area() - obj.m_bboxes[ind - 1].area();
		areaDiff = abs(areaDiff);
		float areaRatio = (float)areaDiff / (float)obj.m_bboxes.back().area();

		return areaRatio < 0.3;
	}

	bool CTrack::isStableDetection(const CObject &obj, int len)
	{

		if (obj.m_bboxes.size() < len)
			return false;

		int startInd = obj.m_bboxes.size() - len - 1;
		for (int i = startInd; i < obj.m_bboxes.size(); i++)
			if (obj.m_detectionTypes[i] <= DET_TYPE::Prediction)
				return false;

		return true;

	}
	/*----------------------------------------------------------------------------------------
	 * Shrink MOG area (bbox) according to shadow angle in image
	 * We do that by the clockDirection - were the shadow point (as clock hour):
	 * We DONT touch tracker boxes 
	----------------------------------------------------------------------------------------*/
	void CTrack::removeShadows(float shadowClockDirection)
	{
		// missing >> : calc 'cut' by clockDirection 
		float cut = 0.4; // 0.5 of image from left side
		for (auto &obj : m_objects) {
			if (obj.m_lastDetected == m_frameNum && obj.m_detectionTypes.back() == DET_TYPE::BGSeg) {
				int trimSize = int((float)obj.m_bboxes.back().width * (1. - fabs(cut)));
				obj.m_bboxes.back().width -= trimSize;
				if (cut < 0)
					obj.m_bboxes.back().x += trimSize;
			}
		}
	}

	void CTrack::removeShadows(std::vector<cv::Rect>  &rois, float shadowClockHand, std::vector<LABEL> labels)
	{
		float cut;
		if (shadowClockHand > 1.5  && shadowClockHand < 4.5)
			cut = 0.4;
		else if (shadowClockHand > 7.5  && shadowClockHand < 10.5)
			cut = -0.4;
		else
			return;

		for (int i = 0; i < rois.size();i++) {
			//if (labels[i] == LABEL::HUMAN) 
			{
				int trimSize = int((float)rois[i].width * fabs(cut));
				rois[i].width -= trimSize;
				if (cut < 0)
					rois[i].x += trimSize;
			}
		}
	}

	void CTrack::removeShadow(CObject &obj, float shadowClockDirection)
	{
		cv::Rect sBbox;
		// missing >> : calc 'cut' by clockDirection 
		float cut = 0.4; // 0.5 of image from left side
		//if (obj.m_lastDetected == m_frameNum && obj.m_detectionTypes.back() < STAGE::TRACKED) 
		{
			int trimSize = int((float)obj.m_bboxes.back().width * (1. - fabs(cut)));
			obj.m_bboxes.back().width -= trimSize;
			if (cut < 0)
				obj.m_bboxes.back().x += trimSize;
		}
	}


	std::vector<LABEL>   CTrack::classify(cv::Mat img, cv::Mat bgMask, std::vector <cv::Rect>  rois)
	{
		const float filllingRatio = 0.6;
		std::vector <LABEL> labels;
		LABEL  label;

		// by size
		for (int i = 0; i < rois.size(); i++) {
			label = classify(img, bgMask, rois[i]);
			labels.push_back(label);
		}

		return labels;
	}


	LABEL   CTrack::classify(cv::Mat img, cv::Mat bgMask, cv::Rect  roi)
	{
		const float filllingRatio = 0.6;
		LABEL label;

		if (roi.width < int((float)SIZES::minHumanWidth * 0.5) ||
			roi.height < int((float)SIZES::minHumanHeight * 0.5))
			return 	LABEL::OTHER;
		else if (roi.width <= int((float)SIZES::maxHumanWidth * 0.5) &&
			roi.height <= int((float)SIZES::maxHumanHeight * 0.5) &&
			(float)roi.width / (float)roi.height < 0.8)
			return 	LABEL::HUMAN;
		else // if (roi.width < int((float)SIZES::minVehicleWidth * m_params.scale))
			return 	LABEL::VEHICLE;
	}

	/*--------------------------------------------
	 * Remove tracked area from motion mask
	---------------------------------------------*/
	void CTrack::pruneBGMask(cv::Mat &mask)
	{
		for (auto obj : m_objects) {
			if (obj.m_lastDetected == m_frameNum) {
				cv::Rect debug = resizeBBox(obj.m_bboxes.back(), m_frame.size(), 1.5);
				mask(resizeBBox(obj.m_bboxes.back(), m_frame.size(), 1.5)).setTo(0); // Note: assume mask same size a m_frame
			}
		}
	}


	/*-------------------------------------------------------------------------------------------------------
	 *        DRAW functions 
	 *-------------------------------------------------------------------------------------------------------*/


	int CTrack::draw()
	{
		static int key = 0;

		draw(m_frameOrg, 1. / m_params.scale);

		// DDEBUG DDEBUG SECTION
		if (m_params.debugLevel > 9)
		{
			draw(m_frame, 1.);
			imshow("video", m_frame);

			// if (m_params.record)  demoRecorder.write(display);

			key = cv::waitKey(m_params.waitKeyTime);
			if (m_frameNum == 1) // DDEBUG 
				key = 'p';
			if (key == 'p')
				key = cv::waitKey(-1);

			return key;
		}
		else
			return 0;

	}

	void CTrack::draw(cv::Mat &img, float scale)
	{
		//cv::Mat display = m_frame.clone();
		cv::Mat display = img;

		cv::Point titlePos(50, 60); // frameNum display
		cv::putText(display, std::to_string(m_frameNum), titlePos, cv::FONT_HERSHEY_SIMPLEX, 2., cv::Scalar(0, 200, 50), 2);

		int displayedObjects = 0;
		// Draw detection rects
		for (int i = 0; i < m_objects.size(); i++) {
			if (m_objects[i].m_detectionStatus < STAGE::FINE)
				continue;
			//cv::Scalar color(0, 0, 250, 2);
			cv::Scalar color;
			color = setColorByDetection(m_objects[i].m_detectionTypes.back());
			//color = setColorByDimensions(m_objects[i].m_bboxes.back());
			// color = setColorByLabel(LABEL(m_objects[i].m_label));

			cv::Rect2f bbox = scaleBBox(m_objects[i].m_bboxes.back(), scale);

			//if (m_objects[i].m_ID != 5) // DDEBUG DDEBUG DDEBUG DDEBUG DDEBUG DDEBUG DDEBUG DDEBUG DDEBUG DDEBUG DDEBUG 
			{
				cv::rectangle(display, bbox, color, 4);

				if (m_objects[i].m_detectionTypes.back() == DET_TYPE::Tracker)
					UTILS::drawCorss(display, bbox, cv::Scalar(255, 255, 255));

				cv::Point text_pos = cv::Point(bbox.x, bbox.y - 10);
				if (m_params.debugLevel > 1) // DDEBUG MODE 
					cv::putText(display, std::to_string(m_objects[i].m_ID), text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
				else if (m_params.showTime)
					cv::putText(display, std::to_string(1 + (int)((float)m_objects[i].age() / (float)CONSTANTS::FPS)), text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
			}
			displayedObjects++;

		}

		// print num of detected objects
		titlePos.x += 1600;
		cv::putText(display, "(#" + std::to_string(displayedObjects) + ")", titlePos, cv::FONT_HERSHEY_SIMPLEX, 1.7, cv::Scalar(0, 0, 0), 2);

	}


	int debugCheckOverlapping(std::vector<CObject>   &objects)
	{
		int overlapped = 0;
		for (int i = 0; i < objects.size(); i++)
			for (int j = i + 1; j < objects.size(); j++)
				//if ((objects[i].m_bboxes.back() & objects[j].m_bboxes.back()).area() > 0) {
				if (bboxesBounding(objects[i].m_bboxes.back(), objects[j].m_bboxes.back()) > 0.1) {
					std::cout << "Objets : " << i << " and " << j << " overlapped \n";
					overlapped++;
				}

		return overlapped;

	}

	int debugCheckOverlapping(std::vector<CObject>   &objects, cv::Rect2f roi, float threshold)
	{
		int overlapped = 0;
		for (int i = 0; i < objects.size(); i++)
				if (bboxesBounding(roi, objects[i].m_bboxes.back()) > threshold) {
					std::cout << "Objets : " << i << " overlapped new ROI \n";
					overlapped++;
				}

		return overlapped;

	}

	
#endif

#if 0
	/*-----------------------------------------------------------------------
	Predict by prev motions
	Check that predicted rect overlapped the motion rect
 -----------------------------------------------------------------------*/
	cv::Rect CTrack::predict(CObject obj)
	{
		//int skip = 2; // optimize motion step - increase motion speed
		const int PredictionDepth = CONSTANTS::FPS;  
		const float MinOverLappedRatio = 0.2;
		const int BackStepsForMotion = 10;

		if (obj.m_bboxes.size() < 2)
			return obj.m_bboxes.back();

		int steps = MIN(PredictionDepth, obj.m_bboxes.size() - 1);
		int lastInd = obj.m_bboxes.size() - 1;
		cv::Rect predictBox;


		int predictionMethod = 3; // <<<--------  DDEBUG 

		if (predictionMethod == 1) {
			cv::Point motion;
			std::vector <cv::Point2f> motion2f;
			std::vector <cv::Point2f> accel;

			for (int i = 0; i < steps; i++) {
				motion = centerOf(obj.m_bboxes[lastInd - i]) - centerOf(obj.m_bboxes[lastInd - i - 1]);
				motion2f.push_back(cv::Point2f(motion.x, motion.y));
			}

			for (int j = 0; j < motion2f.size() - 1; j++)
				accel.push_back(motion2f[j + 1] - motion2f[j]);

			cv::Point2f meanMotion;
			cv::Point2f meanVel = std::accumulate(motion2f.begin(), motion2f.end(), cv::Point2f(0, 0)) / (float)motion2f.size();
			cv::Point2f meanAcc(0, 0);
			
			bool useAcceleration = false;
			if (useAcceleration) {
				cv::Point2f meanAcc(0, 0);
				if (!accel.empty())
					meanAcc = std::accumulate(accel.begin(), accel.end(), cv::Point2f(0, 0)) / (float)accel.size();
			}

			meanMotion = meanVel + meanAcc;


			predictBox = obj.m_bboxes[lastInd] + cv::Point((int)meanMotion.x, (int)meanMotion.y);
			//if (meanMotion.x < 1. || meanMotion.y < 1.) ......
		}
		if (predictionMethod == 10) {
			int steps = MIN(PredictionDepth, obj.m_bboxes.size() - 1);
			int lastInd = obj.m_bboxes.size() - 1;

			cv::Point2f motion10steps = centerOf2f(obj.m_bboxes[lastInd]) - centerOf2f(obj.m_bboxes[lastInd - 10]);
			cv::Point2f motion1step = motion10steps / 10.;
			predictBox = obj.m_bboxes[lastInd] + cv::Point(motion1step);
		}
		else 
		{
			// init motion vectors
			vector<double> iData, xData, yData;
			int fromIndex = obj.m_bboxes.size() - steps - 1;
			int ii = fromIndex;
			for (; ii < obj.m_bboxes.size(); ii++) {
				iData.push_back(double(ii));
				xData.push_back(centerOf2f(obj.m_bboxes[ii]).x);
				yData.push_back(centerOf2f(obj.m_bboxes[ii]).y);
			}
			double predictX, predictY;

			if (predictionMethod == 2) {			//boost::math::interpolators::cardinal_cubic_b_spline<double> spline(xData.begin(), f.end(), x0, dx);


				bool extrapolate = true;
				predictX = interpolate(iData, xData, double(ii), extrapolate);
				predictY = interpolate(iData, yData, double(ii), extrapolate);

				//predictBox = moveByCenter(obj.m_bboxes[lastInd], cv::Point((int)predictX1, (int)predictY1));

				//cv::Point2f meanMotion2 = cv::Point((int)predictX, (int)predictY) - centerOf(obj.m_bboxes[lastInd]);
			}
			else if (predictionMethod == 3) {
				std::vector<double> xx;
				xx.push_back(ii);
				std::vector<double> predictXX = interpolation2(iData, xData, xx);
				std::vector<double> predictYY = interpolation2(iData, yData, xx);

				predictX = predictXX[0];
				predictY = predictYY[0];
			}

			predictBox = moveByCenter(obj.m_bboxes[lastInd], cv::Point((int)predictX, (int)predictY));

		}



		return predictBox;
	}

	/*-----------------------------------------------------------------------
		Predict by prev motions
		Check that predicted rect overlapped the motion rect
	 -----------------------------------------------------------------------*/
	 cv::Rect CTrack::predictNext(CObject obj, cv::Rect mogROI, DET_TYPE &type)
	{
		 const int PredictionDepth = 4;
		 const float MinOverLappedRatio = 0.2;
		 if (obj.m_bboxes.size() < 2)
			 return obj.m_bboxes.back();
		 
		 int steps = MIN(PredictionDepth, obj.m_bboxes.size() - 1);
		 int lastInd = obj.m_bboxes.size() - 1;
		 
		 cv::Point motion;
		 std::vector <cv::Point2f> motion2f;
		 std::vector <cv::Point2f> accel;

		 for (int i = 0; i < steps; i++) {
			 motion = centerOf(obj.m_bboxes[lastInd - i]) - centerOf(obj.m_bboxes[lastInd - i - 1]);
			 motion2f.push_back(cv::Point2f(motion.x, motion.y));
			 //motion += centerOf(obj.m_bboxes[lastInd - i]) - centerOf(obj.m_bboxes[lastInd - i - 1]);
		 }

		 for (int i = 0; i < steps-1; i++)
			 accel.push_back(motion2f[i+1] - motion2f[i]);

		 cv::Point2f meanVel = std::accumulate(motion2f.begin(), motion2f.end(), cv::Point2f(0, 0)) / (float)motion2f.size();
		 cv::Point2f meanAcc(0,0);
		 if (!accel.empty())
			meanAcc = std::accumulate(accel.begin(), accel.end(), cv::Point2f(0, 0)) / (float)accel.size();

		 cv::Point2f meanMotion = meanVel + meanAcc;
		 cv::Rect predictBox = obj.m_bboxes[lastInd] + cv::Point((int)meanMotion.x, (int)meanMotion.y);

		 // Require minimal overlapping for none-hidden type 
		 if (bboxesBounding(predictBox, mogROI) > MinOverLappedRatio) {
			 //predictBox = predictBox & mogROI; // Update size with real ROI
			 type = DET_TYPE::Prediction;
		 }
		 else
			 type = DET_TYPE::Hidden;

		return predictBox;
	}

#endif 

#if 0

	 /*---------------------------------------------------------------
	  * Track all given ROI's by Tracker.
	  * Add new tracker by frameNuym within the m_roiList structure
	 ----------------------------------------------------------------*/
	 int 	CTrack::trackByROI(cv::Mat frame)
	 {
		 ////>>>> tracking by ROI file:
		 // DDEBUG : init tracker from file's list 
		 for (auto &roi : m_roiList) {
			 // Add the new tracker
			 if (m_frameNum == roi.frameNum) {
				 CTracker *newTracker = new CTracker;
				 m_trackers.push_back(*newTracker);
				 m_trackers.back().init(m_params.trackerType, 0);
				 m_trackers.back().setROI(frame, roi.bbox);
				 roi.frameNum *= -1; // sign as nonactive 
				 // Add additionla obj data
				 int objLabel = roi.bbox.width > 70 ? 2 : 1; // classifies obj by ROI size + DDEBUG CONST  
				 m_objects.push_back(CObject(roi.bbox, m_frameNum));
				 //roiList.erase(roiList.begin()+ind); // remove from init list 
			 }
		 }

		 int tracked_count = detectByTracker(frame);
		 return tracked_count;
	 }




	 // UNUSED FUNCTIONS :

	 int CTrack::detectByTracker(const cv::Mat &frame)
	 {
		 int tracked_count = 0;
		 int ind = 0;
		 cv::Rect trackerROI;

		 for (auto &obj : m_objects) {
			 if (obj.m_tracker == NULL && obj.m_detectionTypes.back() == DET_TYPE::Tracker) {
				 // Set a new tracker 
				 obj.m_tracker = new CTracker;
				 obj.m_tracker->init(m_params.trackerType, 0);
				 //obj.m_tracker->setROI(frame, resizeBBox(obj.m_bboxes.back(),0.7));
				 obj.m_tracker->setROI(frame, obj.m_bboxes.back());
				 obj.m_tracker->setROI(frame, obj.m_bboxes.back());
			 }
			 else if (obj.isTracked()) {
				 if (obj.m_tracker->track(frame)) {
					 trackerROI = obj.m_tracker->getBBox();
					 UTILS::checkBounderies(trackerROI, frame.size());
					 obj.add(trackerROI, m_frameNum, DET_TYPE::Tracker);
					 //obj.m_lastDetected = m_frameNum;
					 tracked_count++;
				 }
			 }
		 }

		 return tracked_count;
	 }


	 bool CTrack::detectObjByTracker(const cv::Mat &frame, int objInd)
	 {
		 int tracked_count = 0;
		 int ind = 0;
		 cv::Rect trackerROI;
		 auto &obj = m_objects[objInd];

		 if (obj.m_tracker == NULL && obj.m_detectionTypes.back() == DET_TYPE::Tracker) {
			 // Set a new tracker 
			 obj.m_tracker = new CTracker;
			 obj.m_tracker->init(m_params.trackerType, 0);
			 //obj.m_tracker->setROI(frame, resizeBBox(obj.m_bboxes.back(),0.7));
			 obj.m_tracker->setROI(frame, obj.m_bboxes.back());
		 }
		 else if (obj.isTracked()) {
			 if (obj.m_tracker->track(frame)) {
				 trackerROI = obj.m_tracker->getBBox();
				 UTILS::checkBounderies(trackerROI, frame.size()); // ??? DDEBUG 
				 obj.add(trackerROI, m_frameNum, DET_TYPE::Tracker);
				 //obj.m_lastDetected = m_frameNum;
				 tracked_count++;
			 }
			 else
				 return false;
		 }

		 return true;
	 }

	 /*--------------------------------------------------------------
	  * Detect all tracked obj
	  --------------------------------------------------------------*/
	 int CTrack::detectByOpticalFlow(const cv::Mat &frame)
	 {
		 int detected = 0;
		 for (int i = 0; i < m_objects.size(); i++)
			 if (detectObjByOpticalFlow(frame, i))
				 detected++;

		 return detected;
	 }


	 /*--------------------------------------------------------------
	  * Detect a single tracked obj
	  --------------------------------------------------------------*/
	 bool CTrack::detectObjByOpticalFlow(const cv::Mat &frame, int objInd)
	 {
		 int tracked_count = 0;
		 int ind = 0;
		 cv::Rect trackerROI;
		 auto &obj = m_objects[objInd];

		 if (obj.m_motionTracker == NULL && obj.m_detectionTypes.back() == DET_TYPE::OpticalFlow) {
			 // Set a new tracker 
			 obj.m_motionTracker = new CMotionTracker;
			 obj.m_motionTracker->init(frame.cols, frame.rows);
		 }
		 else  if (obj.isMTracked()) {
			 if (obj.m_motionTracker->track(frame, obj.m_bboxes.back())) {
				 trackerROI = obj.m_motionTracker->getBBox();
				 obj.add(trackerROI, m_frameNum, DET_TYPE::OpticalFlow);
				 //obj.m_lastDetected = m_frameNum;
			 }
			 else
				 int debug = 10;
		 }

		 return obj.m_lastDetected == m_frameNum;
	 }


#endif 
#if 0
	 {

		 // -1- Matched by boxes Similarity 
		 //----------------------------------
		 if (bboxRatio(newROIs[i], oldObj.m_bboxes.back()) > 0.7) { // two boxes over-lappeded and with similar size 
			 newBox = newROIs[i]; // match by box similarity 
			 detectionType = DET_TYPE::BGSeg;
			 match[i]++; // Allow multiple matching 
		 }

		 // -2- Matched by Tracker (Optical)
		 //----------------------------------
		 else if (m_params.onlineTracker > 0 && oldObj.m_detectionStatus >= STAGE::STABLE &&
			 !UTILS::nearEdges(m_frame.size(), oldObj.m_bboxes.back())) {

			 static CMotionTracker trackerOptical;
			 cv::Rect trackBox = trackerOptical.track(m_frame, m_prevFrame, resizeBBox(oldObj.m_bboxes.back(), m_frame.size(), 2.0));
			 if (!trackBox.empty()) {
				 newBox = moveByCenter(oldObj.m_bboxes.back(), centerOf(trackBox)); // use trtacker box just for its center 
				 detectionType = DET_TYPE::OpticalFlow;
			 }
		 }

		 // -3- Otherwise : Matched by Prediction
		 //----------------------------------------
		 if (detectionType == DET_TYPE::DETECTION_NA && oldObj.m_detectionStatus >= STAGE::SENIOR) {
			 newBox = predictNext(oldObj, newROIs[i], detectionType); // match by prediction 
			 match[i]++;
		 }

		 oldObj.add(newBox, m_frameNum, detectionType);
		 break; // loop termination state 						
	 }
				}
			}
			objInd++;
		}
#endif 

#if 0
		/*-----------------------------------------------------------
		 * Match RECTs to exiting m_objects
		 * If newROI matched update BBOX
		 * unless this bbox already updated (by tracker)
		 * If no match was found:  Add a new to m_objects
		 * Return number of new objects
		 *-----------------------------------------------------------*/
		int CTrack::matchObjects_OLD(std::vector<cv::Rect> newROIs, std::vector<LABEL> labels)
		{
			/*-------------------------------------------
			 * match new objects to the new ones
			 * Check box overlapping to match new & old
			 -------------------------------------------*/
			std::vector <int> match(newROIs.size(), 0);
			//match.assign(newROIs.size(), 0);
			for (int i = 0; i < newROIs.size(); i++) {
				for (auto &oldObj : m_objects)
					if ((newROIs[i] & oldObj.m_bboxes.back()).area() > 0 &&
						bboxRatio(newROIs[i], oldObj.m_bboxes.back()) > 0.7) { // DANGEROUSE? 
						if (oldObj.m_lastDetected != m_frameNum) {	// already detected by other algo (tracker etc.)
							oldObj.add(newROIs[i], m_frameNum);
							match[i] = 1; // Mark this ROI as matched
							break;
						}

					}
			}

			// A new objects if no match found
			for (int i = 0; i < match.size(); i++) {
				if (match[i] == 0) {
					m_objects.push_back(CObject(newROIs[i], m_frameNum));
					//m_objects.back().m_label = labels[i];
				}
			}

			return (int)match.size() - cv::countNonZero(match);

		}
#endif 

#if 0
		int CTrack::processFrame(cv::Mat &frame)
		{
			/*
			// -1- Track by object's builtin tracker
			//----------------------------------------
			if (m_params.trackerType > 0) {
				if (m_params.trackerType == 10)
					tracked_count = detectByOpticalFlow(frame);
				else
					tracked_count = detectByTracker(frame);
			}
	*/

	// -2- Find new motion by MOG
	//------------------------------------
	//if (m_frameNum % m_params.detectionFPS == 0) 
			cv::Mat sFrame;
			//cv::resize(frame, sFrame, cv::Size(0, 0), m_calcScale2, m_calcScale2);
			sFrame = frame;

			m_bgMask = m_bgSeg.process(sFrame);
			if (m_params.debugLevel > 1 && !m_bgMask.empty()) {
				cv::imshow("mog org", m_bgMask);
				//m_bgMask.setTo(0, m_bgMask < 255); // = when shadow filter is on 
				if (m_params.trackerType > 0) {
					pruneBGMask(m_bgMask); // remove tracked ROIs from motion mask
					cv::imshow("mog pruned", m_bgMask);
				}
			}

#if 0
			// Canny
			if (0)
			{
				cv::Mat bFrame, edgeImg;
				cv::blur(sFrame, bFrame, cv::Size(3, 3));
				int canny_low_threshold = 30;
				cv::Canny(bFrame, edgeImg, canny_low_threshold, 3 * canny_low_threshold, 3);
				cv::imshow("Canny", edgeImg);
			}
#endif 

			// -3- Find  motion's blobs (objects)
			//------------------------------------
			if (!m_bgMask.empty()) {
				//std::vector<cv::KeyPoint> myBlobs = detectBySimpleBlob(m_bgMask);
				std::vector<cv::Rect>    newROIs = detectByContours(m_bgMask);

				//std::vector<LABEL>   labels = classify(sFrame, m_bgMask, newROIs);
				if (m_params.shadowclockDirection > 0)
					removeShadows(newROIs, m_params.shadowclockDirection, std::vector<LABEL>());

				std::vector <int> matchVec = matchObjects(newROIs); // match blobs to existing objects 

				// *** Create A NEW OBJECT if no match found
				// ***-------------------------------------------***
				for (int i = 0; i < matchVec.size(); i++) {
					if (matchVec[i] == 0) {
						float newROIoverLappedRatio = 0;

						if (!debugCheckOverlapping(m_objects, newROIs[i], 0.1)) {
							if (m_objectID_counter == 8)
								int debug = 10;
							m_objects.push_back(CObject(newROIs[i], m_frameNum, ++m_objectID_counter));
							m_predictions.push_back(CPredict());

							/*
							// check for overlapping
							for (int o = 0; o < m_objects.size(); o++) {
								//overLappedRatio = OverlappingRatio(newROIs[i], m_objects[o].m_bboxes.back());
								newROIoverLappedRatio = bboxesBounding(newROIs[i], m_objects[o].m_bboxes.back());

								if (newROIoverLappedRatio > 0.1) // DDEBUG CONST
									break; // Don't make a new Obj from ROI in case its overlapped other obj
							}

							if (newROIoverLappedRatio < 0.1) { // Don't make a new Obj from ROI in case its overlapped other obj
								if (debugCheckOverlapping(m_objects, newROIs[i]))
									int debug = 10;// DDEBUG
								m_objects.push_back(CObject(newROIs[i], m_frameNum, ++m_objectID_counter));
								m_predictions.push_back(CPredict());
							}
						*/

						}
					}
				}
			}
			/*
					// OLD tracker - now in Match() function
					if (m_params.trackerType > 0) // optical flow
						for (int i = 0; i < m_objects.size(); i++) {
							if (m_objects[i].m_lastDetected < m_frameNum)  // try tracking if was not detected by mog
							{
								if (m_params.trackerType == 10)
									detectObjByOpticalFlow(frame, i);
								else
									detectObjByTracker(frame, i);
							}
						}
			*/

			// -3- consolidate detection
			//------------------------------------
			consolidateDetection();

			//trackByROI(frame);

			m_prevFrame = m_frame.clone();

			return tracked_count;
		}

#endif 