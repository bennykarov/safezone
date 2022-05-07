#ifndef CONCLUDER_HPP
#define CONCLUDER_HPP

#pragma once

#include "types.hpp"


#define CLOSE_ENOUGH  30 // pixels

cv::Point2f center(cv::Rect r); 
/*
template <typename T>
double  distance(T p1, T p2)  { return norm(p1-p2); }
*/



class CConcluder {
public:
	void init();
	void add(std::vector <YDetection> output);
	int  process();
    int size() { return m_objects.size();}

	CObject2 get(int i) {  return m_objects[i]; }

private:
    int match(YDetection detecion);

private:
	std::vector <CObject2> m_objects;


private:


};

#endif 