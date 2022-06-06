#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#include <fstream>

#include <opencv2/opencv.hpp>
#include "yolo5.hpp"
#include "trackerBasic.hpp"
#include "concluder.hpp"



// ================  UTILS ==========================================
const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};


std::string GetCurrentWorkingDir( void ) 
{
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}


    enum Classes
    {
        person = 0,
        bicycle,    //   1
        car,        //   2
        motorbike,  //   3
        aeroplane,  //   4
        bus,        //   5
        train,      //   6  
        truck       //   7
    };


std::vector  <cv::Rect> splitImage(cv::Size size, int pieces_1D)
{
    std::vector  <cv::Rect> ROIs;
    int piece_w = int((float)size.width / (float)pieces_1D);
    int piece_h = int((float)size.height / (float)pieces_1D);

    for (int i=0;i<pieces_1D;i++) {
        int x = piece_w * i; 
        for (int j=0;j<pieces_1D;j++) {
            int y = piece_h * j; 
            ROIs.push_back(cv::Rect(x,y,piece_w, piece_h));
        }
    }
    

    // Check borders
    if (ROIs.back().x + ROIs.back().width >= size.width)
        ROIs.back().x =  size.width - ROIs.back().width - 1;
    if (ROIs.back().y + ROIs.back().height >= size.height)
        ROIs.back().y =  size.height - ROIs.back().height - 1;

        return ROIs;
}


class CMotion {
    public:
    void detect(cv::Mat frame) {}
};

int main(int argc, char **argv)
{
    CMotion _tracker;
    CYolo5 _yolo;
    CConcluder _concluder;
    int split = 0;
    std::vector <cv::Rect> objROIs;

    bool firstTime = true;

    // Trick for move back from  "build" folder by VSCODE make: 
    //std::cout << "Current dir = " << GetCurrentWorkingDir() << "\n";
    std::string curPath = GetCurrentWorkingDir();
    if (curPath.find("/build") != std::string::npos)
        chdir("../");

    std::string inputFName = "/media/bennyk/hddData/bennyk/src/ML/YOLO/yolov5-opencv-cpp-python-main/sample.mp4";
    //std::string inputFName = "/media/bennyk/ssdData/Bautech/rami_levi_111.ts";
    bool is_cuda = true;

    // Get command atguments :
    if (argc > 3) {
        split = atoi(argv[3]);
        std::cout << "split image by" << split*split << "\n";
    }
    if (argc > 2) 
         is_cuda = strcmp(argv[2], "cuda") == 0;
    if (argc > 1) 

        inputFName = argv[1];
        
    // Init
    if (!_yolo.init(is_cuda)) {
        std::cout << "Cant init YOLO5 net , quit \n";
        return -1;
    }


    // Tracker
    int trackerType = 0;
    CTracker *tracker = new CTracker;
    tracker->init(trackerType, 0);

    cv::Mat frame;
    cv::VideoCapture capture(inputFName);
    if (!capture.isOpened())  {
        std::cerr << "Error opening video file\n";
        return -1;
    }

    int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;
    int waitTime = -1;
    int frameNum=0;

    std::vector<Detection> output;
    std::vector< std::vector<cv::Point>> tracks;

    _concluder.init();
    

    while (true)
    {
        capture.read(frame);
        if (frame.empty()) {
            std::cout << "End of stream\n";
            break;
        }

        // Step 1: Motion detection
        _tracker.detect(frame);

        // MAIN detection function:
        if (frame_count % 3 == 0)  {
            output.clear();
            if (split > 1)
                { objROIs = splitImage(frame.size() , split); _yolo.detect(frame, output, objROIs); }
            else
                _yolo.detect(frame, output);
        }

        _concluder.add(output);

        frame_count++;
        total_frames++;

        bool displayYolo = true;
        // Dislpay YOLO detection ROIs:
        if (displayYolo) {
            for (int i = 0; i < output.size(); ++i)  {
                auto detection = output[i];
                if (detection.class_id == Classes::person)
                {
                    auto box = detection.box;
                    auto classId = detection.class_id;
                    const auto color = colors[classId % colors.size()];
                    cv::rectangle(frame, box, color, 3);

                    cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
                    cv::putText(frame, _yolo.getClassStr(classId).c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
        }

        if (0)
        {
            _concluder.process();
            // Dislpay FINAL (Concluder) detection ROIs:
            for (int i = 0; i < _concluder.size(); i++)
            {
                CObject obj = _concluder.get(i);
                const auto color = colors[obj.class_id % colors.size()];
                cv::rectangle(frame, obj.box, color, 3);

                cv::rectangle(frame, cv::Point(obj.box.x, obj.box.y - 20), cv::Point(obj.box.x + obj.box.width, obj.box.y), color, cv::FILLED);
                cv::putText(frame, _yolo.getClassStr(obj.class_id).c_str(), cv::Point(obj.box.x, obj.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        if (frame_count >= 30) {

            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {

            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        float display_w = 1280.;
        if (frame.cols > (int)display_w)
            cv::resize(frame, frame, cv::Size(0,0),display_w/(float)frame.cols,display_w/(float)frame.cols); // DDEBUG 

        cv::imshow("output", frame);

        
        int key = cv::waitKey(waitTime);
        waitTime = 1;

        if (key == 27) {
            capture.release();
            std::cout << "finished by user\n";
            break;
        }
        else if (key == ' ')
            waitTime = -1;
    }

    std::cout << "Total frames: " << total_frames << "\n";

    return 0;
}