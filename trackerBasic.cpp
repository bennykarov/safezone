/*---------------------------------------------------------------------------------------------
 * Best RT tracker : CSRT (in opencv) or
 * the actual winner of VOT cha;;ange is : is SiamFC
 *---------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
//#include <opencv2/tracking/tracker.hpp>
#include  "F:\libraries\opencv-4.5.4\build\install\include\opencv2\tracking\tracking.hpp"
#include  "F:\libraries\opencv-4.5.4\build\install\include\opencv2\tracking\tracking_legacy.hpp"
//tracking_legacy.hpp
#include <opencv2/core/ocl.hpp>

#include "utils.hpp"
#include "config.hpp"
#include "trackerBasic.hpp"


using namespace cv;
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()




bool CTracker::init(int type, int debugLevel, int badFramesToReset)
{

	m_trackerType = type;
	m_badFramesToReset = badFramesToReset;
	m_debugLevel = debugLevel;

	std::cout <<  "Tracker type = " << m_trackerTypes_str[m_trackerType] << "\n";


        if (m_trackerType == MIL) // 1
        	m_tracker = TrackerMIL::create();
        else if (m_trackerType == KCF) { // 2
/*
TrackerKCF::Params::Params()
{
      detect_thresh = 0.5f;
      sigma=0.2f;
      lambda=0.0001f;
      interp_factor=0.075f;
      output_sigma_factor=1.0f / 16.0f;
      resize=true;
      max_patch_size=80*80;
      split_coeff=true;
      wrap_kernel=false;
      desc_npca = GRAY;
      desc_pca = CN;

      //feature compression
      compress_feature=true;
      compressed_size=2;
      pca_learning_rate=0.15f;
  }
*/

        	// Customize params:
        	TrackerKCF::Params param;
        	param.desc_pca = TrackerKCF::GRAY | TrackerKCF::CN;
        	/*
			param.desc_npca = 0;
        	param.compress_feature = true;
        	param.compressed_size = 2;
			*/

        	m_tracker = TrackerKCF::create(param);
        	//  tracker->setFeatureExtractor(sobelExtractor);
        }
        else if (m_trackerType == GOTURN) // 5
        	m_tracker = TrackerGOTURN::create();
        else if (m_trackerType == CSRT) {

        	//m_tracker = TrackerCSRT::create(); //6 
            cv::TrackerCSRT::Params  pars;
            float threshold = pars.psr_threshold;
            pars.psr_threshold = 0.055f; // default is : 0.35 
            m_tracker = TrackerCSRT::create(pars); //6
        }
#if 0
		else if (m_trackerType == BOOSTING) // 0
			m_tracker = legacy::TrackerBoosting::create();
		else if (m_trackerType == TLD) // 3
			m_tracker = legacy::TrackerTLD::create();
		else if (m_trackerType == MEDIANFLOW)
			m_tracker = legacy::TrackerMedianFlow::create(); // 4

#endif 
        else
        	return false;

        return true;
}


/*-----------------------------------------------------------------------------------------
 	 Reset tracker - init for the tracker after first initializing
------------------------------------------------------------------------------------------*/

bool CTracker::init()
{
	if (m_trackerType < 0) {
		std::cout << "Missing first Tracker init \n";
		return false;
	}

	if (!m_tracker.empty())
		m_tracker.release();

	/*
	if (m_trackerType == BOOSTING)
		m_tracker = legacy::TrackerBoosting::create();
	else if (m_trackerType == TLD)
		m_tracker = legacy::TrackerTLD::create();
	else if (m_trackerType == MEDIANFLOW)
		m_tracker = legacy::TrackerMedianFlow::create();
	*/
	else if (m_trackerType == MIL)
		m_tracker = TrackerMIL::create();
	else if (m_trackerType == KCF) {
		// Customize params:
		TrackerKCF::Params param;
		param.desc_pca = TrackerKCF::GRAY | TrackerKCF::CN;
		m_tracker = TrackerKCF::create(param);
		//  tracker->setFeatureExtractor(sobelExtractor);
	}
	else if (m_trackerType == GOTURN)
		m_tracker = TrackerGOTURN::create();
	else if (m_trackerType == CSRT)
		m_tracker = TrackerCSRT::create();
	else
		return false;

	return true;
}



bool CTracker::track(cv::Mat frame)
{

	if (m_bbox.width == 0 || m_bbox.height == 0) {
		reset();
		return false;
	}

	// Update the tracking result
	bool ok = m_tracker->update(frame, m_bbox);

	// Manage end-of-life of tracker
	if (ok)
		falseDetectionLen = 0;
	else {
		falseDetectionLen++;
        std::cout << "Tracker failed (" << falseDetectionLen << ")\n";
    }

	if (m_debugLevel > 0) { // DDEBUG SHOW
		if (ok)
			rectangle(frame, m_bbox, Scalar( 255, 0, 0 ), 2, 1 ); // Tracking success : Draw the tracked object
		else
			putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2); // Tracking failure detected
		cv::Mat display = frame.clone();
		// Display tracker type on frame
		putText(display, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
		cv::putText(display, std::to_string(m_frameNum), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

		imshow("Tracking", display);
	}


	// Check if exceeds bad detection len limits - then cancel tracker
	if (falseDetectionLen >= m_badFramesToReset)
		reset(); // set NOT active
	else
		m_frameNum++;

	return ok;

} 


void CTracker::setROI(const cv::Mat &img, cv::Rect bbox)
{
	UTILS::checkBounderies(bbox, img.size());
	m_bbox = bbox;

    if (m_bbox.x < 0)
        m_bbox.x = 0;
    if (m_bbox.y < 0)
        m_bbox.y = 0;
    if (m_bbox.x + m_bbox.width >= img.cols )
        m_bbox.width = img.cols - m_bbox.x - 1;
    if (m_bbox.y + m_bbox.height >= img.rows )
        m_bbox.height = img.rows - m_bbox.y - 1;
	m_tracker->init(img, m_bbox);
	m_frameNum++;
}


cv::Rect CTracker::setROI_GUI(cv::Mat frame)
{
        	// Define initial boundibg box
            cv::Rect bbox;

        	// Uncomment the line below to select a different bounding box
        	bbox = selectROI(frame, false);

        	// Display bounding box.
        	rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        	imshow("Tracking", frame);

        	//m_tracker->init(frame, m_bbox);

        	return bbox;
}


#if 0
int CTracker::track_main(std::string videoFName, int trackerTypeInd, int skip)
{
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[7] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "CSRT"};
    // vector <string> trackerTypes(types, std::end(types));


    // Create a tracker
    string trackerType = trackerTypes[trackerTypeInd];


    Ptr<Tracker> tracker;
    if (trackerType == "BOOSTING")
    	tracker = TrackerBoosting::create();
    if (trackerType == "MIL")
    	tracker = TrackerMIL::create();
    if (trackerType == "KCF")
    	tracker = TrackerKCF::create();
    if (trackerType == "TLD")
    	tracker = TrackerTLD::create();
    if (trackerType == "MEDIANFLOW")
    	tracker = TrackerMedianFlow::create();
    if (trackerType == "GOTURN")
    	tracker = TrackerGOTURN::create();
    if (trackerType == "CSRT")
    	tracker = TrackerCSRT::create();
    // Read video
    //VideoCapture video("videos/chaplin.mp4");
    VideoCapture video(videoFName);
    
    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
        
    }
    
    // Read first frame
    Mat frame;
    bool ok = video.read(frame);
    

    for (int i=0;i<skip;i++)
    	ok = video.read(frame);


    // Run idle for  selection
    while(video.read(frame)) {
        imshow("Idle", frame);
        int k = waitKey(1);
        if(k == 's')
        	break;
        else if(k == 27)
            return 0;
    }

    // Define initial boundibg box
    Rect2d bbox;
    
    // Uncomment the line below to select a different bounding box
    bbox = selectROI(frame, false);

    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Tracking", frame);
    
    tracker->init(frame, bbox);

    int frameNum = 0;
    while(video.read(frame))
    {
     
        // Start timer
        //double timer = (double)getTickCount();
        frameNum++;
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
        
        // Calculate Frames per second (FPS)
        //float fps = getTickFrequency() / ((double)getTickCount() - timer);
        
        
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
        
        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
        
        // Display FPS on frame
        //putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        cv::putText(frame, SSTR(int(frameNum)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        // Display frame.
        imshow("Tracking", frame);
        // Exit if ESC pressed.
        int k = waitKey(1);
        if (k == 's')
            bbox = selectROI(frame, false);
        else if(k == 27)
            break;

    }
    
    return 0;

}
#endif

