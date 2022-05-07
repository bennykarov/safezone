#pragma once 

#include <boost/circular_buffer.hpp>

#include "trackerBasic.hpp" 
#include "MotionTrack.hpp"


const int  HISTORY_LEN = CONSTANTS::FPS *10;

enum DET_TYPE {
	DETECTION_NA = 0,
	Hidden,   // prediction on blind 
	Prediction, // prediction that overlapped motion area 
	BGSeg,
	Tracker,
	OpticalFlow 
};


enum LABEL {
	HUMAN=1,
	CAR,	// 2
	TRACK,	// 3
	VEHICLE,// 4
	OTHER	// 5
};

class CObject {
	public:
	CObject(cv::Rect2f  r, int frameNum, int id, DET_TYPE detectionType = DET_TYPE::BGSeg)
	{  

#ifdef _DEBUG
		//m_detectionTypes.assign(HISTORY_LEN, DET_TYPE::DETECTION_NA);
		//m_bboxes.assign(HISTORY_LEN, cv::Rect());
#else 
		m_detectionTypes.set_capacity(HISTORY_LEN);
		m_bboxes.set_capacity(HISTORY_LEN);
#endif 

		m_ID = id;
		add(r, frameNum, detectionType);
		/*
		m_bboxes.push_back(r);
		m_detectionTypes.push_back(detectionType);
		m_lastDetected = frameNum;
		*/
		m_birthframe = frameNum;
		classify();
	}

	void add(cv::Rect2f p, int frameNum, DET_TYPE type)
	{
		m_bboxes.push_back(p);
		m_detectionTypes.push_back(type);
		m_lastAdded = frameNum;
		if (frameNum > 0 && m_detectionTypes.back() > DET_TYPE::Hidden)
			m_lastDetected = frameNum;
	}

	int len() { return (int)m_bboxes.size(); } // Note tha m_bboxes is a cyclic buffer
	int age() { return  m_lastAdded - m_birthframe; } // Note tha m_bboxes is a cyclic buffer

	void set(int motionFrames, int motionDist)
	{
		m_motionFrames = motionFrames;
		m_motionDist = motionDist;
	}



	// Add last points to RECT - check for the box  dimensios 
	bool isMove(int dist = 10) 
	{
		if (m_bboxes.size() < m_motionFrames)
			return true;

		cv::Rect motionBox;
		for (int i= (int)m_bboxes.size()-1; i > (int)m_bboxes.size()-m_motionFrames;i--) {
			motionBox = extendBBox(motionBox, centerOf(m_bboxes[i]));
		}		

		if (motionBox.width > dist || motionBox.height > dist)
			return true;
		else
			return false;
	}

	int classify( /*cv::Mat img*/)
	{
		if (m_bboxes.back().width < 70)
			m_label = LABEL::HUMAN;
		else
			m_label = LABEL::VEHICLE;

		return m_label;

	}


	static 	int classify( cv::Rect r)
	{
		if (r.width < 70)
			return LABEL::HUMAN;
		else
			return LABEL::VEHICLE;
	}

	// Warning: opencv Tracker only !
	bool isTracked()
	{
		return m_tracker != NULL && m_tracker->isActive();
	}

	bool isTracked(int frameNum)
	{
		return m_lastDetected == frameNum;
	}

	// Motion Tracker (Optical Flow)
	bool isMTracked()
	{
		return m_motionTracker != NULL;
	}


public:
	unsigned int m_ID = -1;
	int       m_label = 0;
	// History of detection:
#ifdef _DEBUG
	std::vector<cv::Rect2f>   m_bboxes; // DDEBUG for debug
	std::vector<DET_TYPE> m_detectionTypes;
#else
	boost::circular_buffer <cv::Rect2f>   m_bboxes;
	boost::circular_buffer<DET_TYPE> m_detectionTypes;
#endif 

	int   m_birthframe = -1;
	int   m_lastAdded = -1;
	int   m_lastDetected = -1;
	//int   m_lastTracked = -1;
	//DET_TYPE m_detectionType =  DET_TYPE::DETECTION_NA;
	int m_detectionStatus = 0; // detection stability 
	CTracker *m_tracker = NULL;
	CMotionTracker *m_motionTracker = NULL;

private:
	int m_motionFrames = 1*CONSTANTS::FPS; // DDEBUG CONST 
	int m_motionDist;

};

