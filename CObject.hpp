#pragma once 

#ifndef COBJECT_HEADER
#define COBJECT_HEADER
#include <boost/circular_buffer.hpp>
#include "config.hpp"


/* OLD OBJECT CLASS
class CObject2
{
public:

	int class_id;
	float confidence;
	cv::Rect box;
	std::vector <cv::Point2f> centers;
};
*/


enum Labels
{
	nonLabled = -1,
	person,		//	 0
	bicycle,    //   1
	car,        //   2
	motorbike,  //   3
	aeroplane,  //   4
	bus,        //   5
	train,      //   6  
	truck       //   7
};


enum DETECT_TYPE {
	DETECT_NA = 0,
	BGSeg,
	ML,
	Tracker,
	Hidden,   // prediction on blind 
	Prediction 
	};


class CObject {
public:
	CObject() :m_bbox(0,0,0,0) { }
	CObject(cv::Rect  r, int frameNum, int id, DETECT_TYPE  detectionType, Labels label)
	{
		m_label = label;
		m_detectionType = detectionType;
		m_frameNum = frameNum;
		m_bbox = r;
	}

	bool empty() { return m_bbox.width == 0; }

public:
	unsigned int	m_ID = -1;
	Labels			m_label = Labels::nonLabled; // current detection label
	Labels			m_finalLabel = Labels::nonLabled;  // Conclusion of all detection labels
	cv::Rect		m_bbox; // DDEBUG for debug
	DETECT_TYPE		m_detectionType;
	int				m_frameNum;

private:
};

#endif 