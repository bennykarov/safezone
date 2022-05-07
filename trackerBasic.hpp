#pragma once
 
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>


#define MAX_BAD_DETECTION_SEC 1 // 4

enum TRACK_TYPE {
    BOOSTING = 0,
	MIL, 		// 1
	KCF, 		// 2
	TLD, 		// 3
	MEDIANFLOW, // 4
	GOTURN,  	// 5
	CSRT,  		// 6
	TYPES_LEN  
};

class CTracker {
public:
	bool init(int TrackerType, int debugLevel, int badFramesToreset = MAX_BAD_DETECTION_SEC * CONSTANTS::FPS);
	bool init();
	void reset()
		{ m_frameNum = 0; falseDetectionLen = 0; m_bbox = cv::Rect(); init();}
	bool  track(cv::Mat frame);
	bool  track(cv::Mat frame, cv::Rect roi);
	void setROI(const cv::Mat &img, cv::Rect bbox);
	bool isActive() { return m_frameNum > 0; }  
	bool isDetected() { return falseDetectionLen == 0;} // currently detected
	// void setDebugLevlel(int l) { m_debugLevel = l;}

	static cv::Rect setROI_GUI(cv::Mat img); // DDEBUG function

	cv::Rect getBBox() { return m_bbox; }

	int getFlaseDetectionLen() { return falseDetectionLen; }
	int track_main(std::string videoFName, int trackerTypeInd, int skip=0);

private:
	// 										0		   1	  2      3       4           5         6
    std::string m_trackerTypes_str[7] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "CSRT"};
    // vector <string> trackerTypes(types, std::end(types));
    int m_trackerType = -1;
    std::string trackerType;
    int m_frameNum = 0; 			// tracking processed frames
	cv::Rect m_bbox;
	int falseDetectionLen = 0; 	//count detection failure tail length

	int m_debugLevel=0;

	int m_badFramesToReset = MAX_BAD_DETECTION_SEC * CONSTANTS::FPS;

    cv::Ptr<cv::Tracker> m_tracker;


};
