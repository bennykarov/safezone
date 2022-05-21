#include <stdio.h>
#include <vector>
#include <algorithm>

#include "opencv2/opencv.hpp"


#include "CObject.hpp"
#include "yolo/yolo5.hpp"

#include "utils.hpp"
#include "concluder.hpp"


void CConcluder::init() { 	m_active = true; }

void CConcluder::add(std::vector <cv::Rect>  BGSEGoutput, std::vector <YDetection> YoloOutput, int frameNum)
{
	if (!m_active)
		return;

	std::vector <int> matchInds;

	m_frameNum = frameNum;
	// Add the new BGSeg object - match to current BGSeg objects (if does)
	// Ignore those were matched with YOLO objects 
	//-------------------------------------------------------------------------
	for (auto bgRect : BGSEGoutput) {
		CObject newObj(bgRect, frameNum, 0, DETECT_TYPE::BGSeg, Labels::nonLabled);  // 	CObject(cv::Rect  r, int frameNum, int id, DETECT_TYPE  detectionType, Labels label)
		int ind = bestMatch(bgRect, 0.2);// , matchInds);
		if (ind >= 0) {
			m_objects[ind].push_back(newObj);
			matchInds.push_back(ind);
		}
		else {
			// New object
			m_objects.push_back(std::vector <CObject>());
			m_objects.back().push_back(newObj);
			//m_goodObjects.push_back(newObj);
		}

		//m_objects.back().back().m_finalLabel = calcFinalLable(m_objects.back());
	}

	// Add YOLO object - match to BGSeg objects (if does)
	//----------------------------------------------------
	for (auto Yobj : YoloOutput) {
		CObject newObj(Yobj.box, frameNum, 0, DETECT_TYPE::ML, (Labels)Yobj.class_id);  // 	CObject(cv::Rect  r, int frameNum, int id, DETECT_TYPE  detectionType, Labels label)

		int ind = bestMatch(Yobj.box, 0.5);
		if (ind >= 0) {
			m_objects[ind].push_back(newObj);
			matchInds.push_back(ind);
		}
		else {
			// New object
			m_objects.push_back(std::vector <CObject>());
			m_objects.back().push_back(newObj);
		}

		//m_objects.back().back().m_finalLabel = calcFinalLable(m_objects.back());
	}

	int rem = pruneObjects();
	if (rem > 0)
		std::cout << "Number of pruned (elders) object = " << rem << "\n";  //int debug = 10;

}

void CConcluder::add(std::vector <YDetection> YoloOutput, int frameNum)
{
}



int CConcluder::bestMatch(cv::Rect box, float overlappedRatio, std::vector <int> ignoreInds)
{
	std::vector <int>  bestInds;
	std::vector <float>  bestScores;

	for (int i = 0; i < m_objects.size(); i++) {
		if (std::find(ignoreInds.begin(), ignoreInds.end(), i) != ignoreInds.end())
			continue; // Ignore this object
		for (auto obj : m_objects[i]) {
			float overlappingRatio = bboxesBounding(obj.m_bbox, box); // most new box overlapped old box
			// Check overlapping ratio and that size are similars (kind of)
			if (overlappingRatio > overlappedRatio && similarAreas(obj.m_bbox, box, 0.6*0.6)) {
				bestInds.push_back(i);
				bestScores.push_back(overlappingRatio);
			}
		}
	}

	if (bestScores.empty())
		return -1;

	auto it = max_element(bestScores.begin(), bestScores.end());
	int index = it - bestScores.begin();
	return bestInds[index];
}

int CConcluder::track() { return 0; }

#if 0
/*---------------------------------------------------------------------------
	Return only "good" objects - with Label or long enough (with no label)
	A good object is:
	(1) Detected on current (last) frame
	(2) Detected by YOLO or is long enough (by any kind of detection)
 ---------------------------------------------------------------------------*/
std::vector <CObject> CConcluder::getObjects_(int frameNum)
{

	for (auto obj : m_objects) {
		obj.back().m_finalLabel = obj.back().m_label =  calcFinalLable(obj);

		if (obj.back().m_frameNum == frameNum &&
			(obj.back().m_finalLabel != Labels::nonLabled || obj.size() >= GOOD_TRACKING_LEN))
			m_goodObjects.push_back(obj.back());
	}

	return goodObjects;
}
#endif 

/*---------------------------------------------------------------------------
	Get object : person labeled and stable BGSeg  
 ---------------------------------------------------------------------------*/
std::vector <CObject> CConcluder::getObjects(int frameNum)
{
	m_goodObjects.clear(); // TEMP clearing

	for (auto &obj : m_objects) {
		CObject consObj = consolidateObj(obj);
		if (!consObj.empty())
			m_goodObjects.push_back(consolidateObj(obj));
	}


#if 0
	// Smooth rect dimensions (by history sizes):
	{
		cv::Size avgDim(0, 0);
		for (auto &obj : m_objects) {
			for (int i = obj.size() - CONCLUDER_CONSTANTS::GOOD_TRACKING_LEN - 1; i < obj.size(); i++) {
				avgDim.width += obj[i].m_bbox.width;
				avgDim.height += obj[i].m_bbox.height;
			}
			avgDim.width += obj.back().m_bbox.width; // double inut (weight) to last rect 
			avgDim.height += obj.back().m_bbox.height;

			// Avrg
			avgDim.width = (int)(float)avgDim.width / (float)(CONCLUDER_CONSTANTS::GOOD_TRACKING_LEN + 1);
			avgDim.height = (int)(float)avgDim.height / (float)(CONCLUDER_CONSTANTS::GOOD_TRACKING_LEN + 1);

			obj.back().m_bbox = resize(obj.back().m_bbox, avgDim);
		}
	}
#endif 

	return m_goodObjects;
}


/*-----------------------------------------------------------------------------------------
 * Get all objects that are NOT person or BGSeg - all other Labeles objects
 * "onlyMoving" - return only objects in motion  
 -----------------------------------------------------------------------------------------*/
std::vector <CObject> CConcluder::getObjectsOthers(int frameNum, bool onlyMoving)
{
	std::vector <CObject> otherObjects;

	for (auto obj : m_objects) {
		if (obj.back().m_finalLabel != Labels::nonLabled && obj.back().m_finalLabel != Labels::person)
			if (!onlyMoving || isMoving(obj))
				otherObjects.push_back(obj.back());
	}

	return otherObjects;
}


std::vector <int> CConcluder::getObjectsInd(int frameNum)
{
	std::vector <int> goodObjectsInd;

	for (int i = 0; i< m_objects.size();i++) {
		if (m_objects[i].back().m_frameNum == frameNum &&
			(m_objects[i].back().m_finalLabel != Labels::nonLabled || m_objects[i].size() >= CONCLUDER_CONSTANTS::GOOD_TRACKING_LEN))
			goodObjectsInd.push_back(i);
	}

	return goodObjectsInd;
}

#if 0
std::vector <CObject> CConcluder::getHiddenObjects(int curFrameNum, int backward)
{
	std::vector <CObject> hiddenObjects;

	for (auto obj : m_objects) {
		if (obj.back().m_frameNum < curFrameNum  && obj.back().m_frameNum >= (curFrameNum - backward) &&
			(obj.back().m_finalLabel != Labels::nonLabled || obj.size() >= GOOD_TRACKING_LEN))
			hiddenObjects.push_back(obj.back());
	}

	return hiddenObjects;
}
#endif 

Labels    CConcluder::calcFinalLable(std::vector <CObject> objectList)
{
	for (auto obj : objectList) {
		if (obj.m_label == Labels::person)
			return Labels::person;
	}
	for (auto obj : objectList) {
		if (obj.m_label != Labels::nonLabled)
			return obj.m_label;
	}

	return Labels::nonLabled;

}

#if 0
CObject   CConcluder::consolidateObj_(std::vector <CObject> objectHist)
{
	CObject consObj = objectHist.back();

	if (objectHist.size() < CONSTANTS::StableLen || m_frameNum - objectHist.back().m_frameNum > CONCLUDER_CONSTANTS::MAX_HIDDEN_FRAMES)
		return CObject();

	// Obj detected by BGSEG - inherit the latest YOLO attribute 
	int backward = MIN(CONCLUDER_CONSTANTS::INHERIT_LABEL_LEN, objectHist.size());
	auto startIt = objectHist.end() - backward;

	auto  itPerson = std::find_if(startIt, objectHist.end(), [](CObject s) { return s.m_label == Labels::person; });
	if (itPerson != objectHist.end()) {
	consObj.m_finalLabel = consObj.m_label = itPerson->m_label;
		// Set original YOLO size to BGSeg rect:
		cv::Point tl = objectHist.back().m_bbox.tl();
		consObj.m_bbox = centerBox(centerOf(objectHist.back().m_bbox), cv::Size(itPerson->m_bbox.width, itPerson->m_bbox.height));
		return consObj;
	}

	auto  itClassified = std::find_if(startIt, objectHist.end(), [](CObject s) { return s.m_label != Labels::nonLabled; });
	if (itClassified != objectHist.end()) {
		consObj.m_finalLabel = consObj.m_label = itClassified->m_label;
		//consObj.m_bbox = resize(itClassified->m_bbox, cv::Size(itClassified->m_bbox.width, itClassified->m_bbox.height));
		return consObj;
	}

	// CASE 3:   BGSeg (survivle) object 
	if (objectHist.size() >= CONCLUDER_CONSTANTS::GOOD_TRACKING_LEN) {
		// smooth rect & pos ...
		return 	objectHist.back();
	}

	return CObject();


}
#endif 

/*----------------------------------------------------------------------------------------------
 * Gather all information to a final object:
 * If (even) once in history the obj was labeled - inherit this label (for INHERIT_LABEL_LEN  frames)
----------------------------------------------------------------------------------------------*/
CObject   CConcluder::consolidateObj(std::vector <CObject> &objectHist)
{
	CObject consObj = objectHist.back();

	// Obj detected by BGSEG - inherit the latest YOLO attribute 
	int backward = MIN(CONCLUDER_CONSTANTS::INHERIT_LABEL_LEN, objectHist.size());
	int lastInd = objectHist.size() - backward;

	// -1- Look for any person label
	//---------------------------------
	int i = objectHist.size()-1;
	//auto  itPerson = std::find_if(startIt, objectHist.end(), [](CObject s) { return s.m_label == Labels::person; });
	while (i >= lastInd && objectHist[i].m_label != Labels::person) i--;
	if (i >= lastInd) {
		consObj.m_finalLabel = consObj.m_label = objectHist.back().m_finalLabel = objectHist[i].m_label;
		// Inherit original (latest) rect size 
		cv::Point tl = objectHist.back().m_bbox.tl();
		consObj.m_bbox = centerBox(centerOf(objectHist.back().m_bbox), cv::Size(objectHist[i].m_bbox.width, objectHist[i].m_bbox.height));
		return consObj;
	}

	// -2- Look for any other label
	//---------------------------------
	i = objectHist.size() - 1;
	//auto  itPerson = std::find_if(startIt, objectHist.end(), [](CObject s) { return s.m_label == Labels::person; });
	while (i >= lastInd && objectHist[i].m_label == Labels::nonLabled) i--;
	if (i >= lastInd) {
		consObj.m_finalLabel = consObj.m_label = objectHist.back().m_finalLabel = objectHist[i].m_label;
		/*
		// Inherit original (latest) rect size
		cv::Point tl = objectHist.back().m_bbox.tl();
		consObj.m_bbox = cv::Rect(tl.x, tl.y, objectHist[i].m_bbox.width, objectHist[i].m_bbox.height);
		*/
		return consObj;
	}

	// CASE 3:   BGSeg (stable) object 
	if (objectHist.size() >= CONCLUDER_CONSTANTS::GOOD_TRACKING_LEN) {
		// smooth rect & pos ...
		return 	objectHist.back();
	}

	return CObject();
}


/*----------------------------------------------------------------
 * Utilities 
 -----------------------------------------------------------------*/
bool CConcluder::isMoving(std::vector <CObject> obj)
{
	const int MIN_LEN = 5;  // in frames 
	const double MIN_DISTANCE = MIN_LEN * 2; // in pixels


	if (obj.size() < MIN_LEN)
		return false;

	int dist = distance(centerOf(obj.back().m_bbox), centerOf(obj[obj.size() - MIN_LEN - 1].m_bbox));
	if (dist >= MIN_DISTANCE)
		return true;

	return false;
}

bool CConcluder::isStatic(std::vector <CObject> obj)
{
	const int MIN_LEN = 7;  // in frames 
	const int TOLERANCE = 7;

	if (obj.size() < MIN_LEN)
		return false;

	int dist = distance(centerOf(obj.back().m_bbox), centerOf(obj[obj.size() - MIN_LEN - 1].m_bbox));
	if (dist > TOLERANCE)
		return false;

	return false;
}

inline bool CConcluder::isLarge(std::vector <CObject> obj)
{
	return (obj.back().m_bbox.width > m_maxPresonDim.width || obj.back().m_bbox.height > m_maxPresonDim.height);
}



int CConcluder::pruneObjects()
{
	int orgSize = m_objects.size();
	// Remove un-detected objects (fade)
	for (int i = 0; i < m_objects.size(); i++) {
		if (m_frameNum - m_objects[i].back().m_frameNum > CONCLUDER_CONSTANTS::MAX_HIDDEN_FRAMES)
			m_objects.erase(m_objects.begin() + i--);
	}

	return orgSize - m_objects.size();
}