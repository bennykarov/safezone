#pragma once 

class CObject2
{
    public:

    int class_id;
    float confidence;
    cv::Rect box;
    std::vector <cv::Point2f> centers;
};
