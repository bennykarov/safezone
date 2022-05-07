#pragma once 
#include <Windows.h>
#include <stdint.h>

typedef struct ALGO_DETECTION_OBJECT_DATA
{
	int X;
	int Y;
	int Width;
	int Height;
	int CountUpTime;
	int ObjectType; // optional
	int DetectionPercentage;// optional
	int reserved1;
	int reserved2;
} ALGO_DETECTION_OBJECT_DATA;

typedef enum BAUOTECH_AND_BENNY_KAROV_ALGO
{
	ALGO_RAMI_LEVI = 2001,
	ALGO_ZOSMAN = 2002,
	ALGO_DEFAULT = 2003,
	ALGO_EMBOSS = 2004,
	ALGO_GREY = 2005,
	ALGO_BLUR = 2006,
	ALGO_POSTERIZE = 2007,
	ALGO_XOR = 2008,
	ALGO_DARKEN = 2009,
	ALGO_BLUE = 2010,
	ALGO_GREEN = 2011,
	ALGO_RED = 2012
} BAUOTECH_AND_BENNY_KAROV_ALGO;




 


#ifdef __cplusplus
extern "C" {


	 

#define API_EXPORT __declspec(dllexport)
#endif


	API_EXPORT int RunAlgoColors(BAUOTECH_AND_BENNY_KAROV_ALGO algo, 
								 uint8_t *pData, 
								 uint32_t width, 
							  	 uint32_t height, 
								 uint32_t image_size,
							     uint8_t youDraw,
								 ALGO_DETECTION_OBJECT_DATA *pObjects,
								 uint32_t *objectCount);
	 


#ifdef __cplusplus
}
#endif
