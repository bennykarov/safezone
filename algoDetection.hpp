#pragma once

#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"

#include "mog.hpp"
#include "prediction.hpp"
#include "CObject.hpp"

#include "yolo/types.hpp"
#include "yolo/yolo5.hpp"
#include "yolo/concluder.hpp"


class CRoi2frame {
public:
	cv::Rect   bbox;
	int frameNum;
};


enum STAGE {
	AGE_1=9999,
	BORN = 0,
	STARTER, // 1,
	FINE,	// 2
	SENIOR, // 3 - once was STABLE 
	STABLE // 4
};


class CDetector {
public:
	bool init(int w, int h, int imgSize, bool isCuda, float scaleDisplay = 0.5);
	int process(void *dataTemp, ALGO_DETECTION_OBJECT_DATA *pObjects);
	int processFrame(cv::Mat &frame);
	void draw(cv::Mat &img, std::vector<YDetection> Youtput, float scale);   // for Yolo
	void draw(cv::Mat &img, std::vector<cv::Rect>  rois, float scale);		 // for BGSeg
	void drawInfo(cv::Mat &img);		

	//int draw();

private:
	bool motionDetected(cv::Mat mask);
	/*
	std::vector<cv::KeyPoint> detectBySimpleBlob(cv::Mat img);
	std::vector <cv::Rect> detectByContours(cv::Mat bgMask);
	int detectByTracker(const cv::Mat &frame);
	int detectByOpticalFlow(const cv::Mat &frame);
	bool detectObjByOpticalFlow(const cv::Mat &frame, int objInd);
	bool detectObjByTracker(const cv::Mat &frame, int objInd);
	
	int detectByTracker_OLD(cv::Mat frame);

	bool isStable(const CObject &obj, int len);
	bool isStableDetection(const CObject &obj, int len);

	std::vector <int> matchObjects(std::vector<cv::Rect> newROIs);
	bool matchByPrediction(int objID, cv::Rect2f mogBox);

	int matchObjects_OLD(std::vector<cv::Rect> newROIs, std::vector<LABEL> lables);
	void consolidateDetection();
	void removeShadows(float shadowClockDirection);
	void removeShadows(std::vector<cv::Rect>  &newROIs, float shadowClockDirection, std::vector<LABEL> labels);
	void removeShadow(CObject &obj, float shadowClockDirection);
	void pruneBGMask(cv::Mat &mask); // Remove tracked area from motion mask
	//void classify();
	std::vector<LABEL>   classify(cv::Mat img, cv::Mat bgMask, std::vector <cv::Rect>  rois);
	LABEL   classify(cv::Mat img, cv::Mat bgMask, cv::Rect  rois);
	cv::Rect predict(CObject obj);
	cv::Rect predictNext(CObject obj, cv::Rect overlappedROI, DET_TYPE &type);

	int trackByROI(cv::Mat frame);
	*/
	std::vector <cv::Rect> detectByContours(cv::Mat bgMask);


private:
	int m_width = 0;
	int m_height = 0;
	void *m_data = NULL;
	int m_frameNum = 0;
	//float m_calcScale = 0.5;
	float m_scaleDisplay = 1.;// 0.7;
	bool  m_motionDetectet = false;


	cv::Mat m_frameOrg; // Original image
	cv::Mat m_frameROI;
	cv::Mat m_frame; // working image
	cv::Mat m_prevFrame; // Prev
	cv::Mat m_bgMask; // MOG2
	cv::Mat m_display;

	CYolo5 m_yolo;
	CConcluder m_concluder;


private:
	int m_doTracking=1;
	int m_doDetection=1;

	std::vector<YDetection> m_Youtput;
	std::vector <cv::Rect>  m_BGSEGoutput;

#if 0
	int status=0;

	// Tracker members
	cv::Rect m_trackerROI;
	// object "class":
	std::vector<CDetector>  m_trackers;
	std::vector<CObject>       m_objects;

	// Detection classes:
	std::vector <CPredict>  m_predictions;
#endif  
	CBGSubstruct   m_bgSeg;
	int m_colorDepth = 4;
	Config m_params;
	std::vector <CRoi2frame>  m_roiList;
	unsigned int m_objectID_counter = 0;
	cv::Rect m_camROI = cv::Rect(0,0,0,0);

};
