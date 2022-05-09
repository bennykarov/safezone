#pragma once


namespace CONSTANTS {
	int const FPS = 30;
	int const DetectionFPS = 3;
	int const motionDetectionFPS = DetectionFPS;
	int const MogEmphasizeFactor = 1;
	int const StableLen = 4;
	int const minLenForPrediction = 10 + 1; // keep odd for acceleration calc
	int const maxLenForPrediction = FPS + 1;  // keep odd for acceleration calc 
	int const StableLenForPrediction = minLenForPrediction+10; // Prediction doesn't usethe first 10 unstable frames

	// NEWs
	int const SKIP_YOLO_FRAMES = 3;

};


namespace  SIZES {
	const int minVehicleWidth = 55 * 2;

	const int minHumanWidth = 15 *2;
	const int maxHumanWidth = 50 * 2;
	const int minHumanHeight = 30 * 2;
	const int maxHumanHeight = 60 * 2;
};


struct Config
{
	// Operational 
	std::string videoName;
	std::string roisName;
	std::string modelFolder;
	int showTime = 1;
	int showBoxesNum = 1;
	int debugLevel = 0;
	int beep = 0;
	float displayScale = 1.;
	// Algo
	int trackerType = 0;
	int MLType = 0;
	int prediction = 1;
	int onlineTracker = 0;
	float scale = 0.5;
	int waitKeyTime=1;
	int record = 0;
	int demoMode=0;
	// MOG2 params:
	int MHistory = 200;
	float MvarThreshold = 20.0;
	float MlearningRate = -1.;
	float shadowclockDirection = 0;
	int detectionFPS = 2;
	std::vector <int> camROI = { 0,0,0,0 }; // RECT 
	};
