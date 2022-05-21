#ifndef CONCLUDER_HPP
#define CONCLUDER_HPP

#pragma once

#include "CObject.hpp"

namespace CONCLUDER_CONSTANTS
{
	const int CLOSE_ENOUGH = 30; // pixels
	const int GOOD_TRACKING_LEN = 20;
	const int INHERIT_LABEL_LEN  = 30 * 1; //  1 sec
	const int MAX_HIDDEN_FRAMES = 4;
}

cv::Point2f center(cv::Rect r);

class CConcluder {
public:
	void init();
	void setPersonDim(cv::Size dim) { m_maxPresonDim = dim;} // max person size in pixels
	void add(std::vector <cv::Rect>  BGSEGoutput, std::vector <YDetection> YoloOutput, int frameNum);
	void add(std::vector <YDetection> YoloOutput,int frameNum);
	int track();
	std::vector <CObject> getObjects(int frameNum); //  { return m_goodObjects; }
	std::vector <CObject> getObjectsOthers(int frameNum, bool only_moving); //  Other labeled objects (not a persons) & BGSeg 
	/*
	std::vector <CObject> getObjects_(int frameNum); // only last detected object
	std::vector <std::vector <CObject>> getObjects(int frameNum);
	*/
	std::vector <int> getObjectsInd(int frameNum);
	std::vector <CObject> getHiddenObjects(int curFrameNum, int backward);

	int size() { return m_objects.size(); }
	/*
	void add(std::vector <YDetection> output);
	void addSimple(std::vector <YDetection> output);
	int  process();
	*/

	CObject get(int i) { return m_objects[i].back(); }
	bool isMoving(std::vector <CObject> obj);
	bool isStatic(std::vector <CObject> obj);
	bool isLarge(std::vector <CObject> obj);


private:
	int match(std::vector <cv::Rect>);
	int match(std::vector <YDetection> detecions);
	//int bestMatch(YDetection Yobj);
	int bestMatch(cv::Rect r, float overlappedRatio, std::vector <int> ignore= std::vector <int>());
	//CObject   consolidateObj_(std::vector <CObject> objectList);
	CObject   consolidateObj(std::vector <CObject> &objectList);
	Labels    calcFinalLable(std::vector <CObject> obj);
	int		  pruneObjects();

private:
	std::vector <std::vector <CObject>> m_objects;
	std::vector <CObject> m_goodObjects;

	int m_idCounter = 0;
	bool m_active = false;
	cv::Size m_maxPresonDim = cv::Size(150 ,200); // DDEBUG CONST 
	cv::Size m_dim;
	int m_frameNum;
};

#endif 