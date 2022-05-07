
#include <opencv2/opencv.hpp>

#include "yolo5.hpp"
#include "concluder.hpp"

// UTILS
cv::Point2f center(cv::Rect r)   { return cv::Point2f((r.br() + r.tl())*0.5);}


void CConcluder::init()
{}

/*----------------------------------------------------------
 *   Match a new detection to one of the (current) objects 
 * Return the matched index
 *----------------------------------------------------------*/
int CConcluder::match(YDetection detection)
{
    return -1; // DDEBUG 
    // First try overlapping rects 
    //-----------------------------------
    int  maxOverlappedArea  = 0; 
    int best_ind = -1;

    for (int i=0;i<m_objects.size();i++) {
        int  overlappedArea  = 0; 
        overlappedArea = (detection.box & m_objects[i].box).area();
        if (maxOverlappedArea < overlappedArea) {
            maxOverlappedArea = overlappedArea;
            best_ind = i;
        }
    }

    if (best_ind > -1)
        return best_ind;


    // Second (no overlapping) - try closer object  
    //---------------------------------------------
    double minDist = 99999.; 
/*
    for (int i=0;m_objects.size();i++) {
        double dist = distance(center(detection.box), m_objects[i].centers.back());
        if (minDist > dist)
            best_ind = i;
    }
*/
    if (minDist <= (double)CLOSE_ENOUGH)
        return best_ind;
    else
        return -1.;

}

void CConcluder::add(std::vector <YDetection> output)
{
    //for (int i = 0; i < output.size(); ++i) {
    for (int i = 0; i < 1; ++i) { // DDEBUG 
        // Missing here: Predict next object data
        int ind = match(output[i]);    
        if (ind < 0) {// unmatch object - make a new obj
            CObject2 newObj;
            newObj.box = output[i].box;
            newObj.class_id = output[i].class_id;
            newObj.confidence = 1;        
            newObj.centers.push_back(center(newObj.box));        
            m_objects.push_back(newObj);
        }
        else {
            // Matched found
            m_objects[ind].box = output[i].box;
            m_objects[ind].class_id = output[i].class_id;
            m_objects[ind].confidence = (m_objects[ind].centers.size() * 10) % 100;        
            m_objects[ind].centers.push_back(center(output[i].box));       
        }
    }


}


int  CConcluder::process()
{
    return 0;
}
