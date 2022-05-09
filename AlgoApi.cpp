#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"

#include "config.hpp"
#include "utils.hpp"
#include "trackerBasic.hpp"
#include "MotionTrack.hpp"

#include "AlgoApi.h"
#include "AlgoDetection.hpp"


CDetector g_tracker;
int frameNum = 0;

// This is an example of an exported function.
API_EXPORT int RunAlgoColors(BAUOTECH_AND_BENNY_KAROV_ALGO algo,
							 uint8_t *pData,
							 uint32_t width,
							 uint32_t height,
							 uint32_t image_size,
							 uint8_t youDraw,
							 ALGO_DETECTION_OBJECT_DATA *pObjects,
							 uint32_t *objectCount)
{
	RGBTRIPLE *prgb;            // Holds a pointer to the current pixel
	int iPixel;                 // Used to loop through the image pixels
	int temp, x, y;               // General loop counters for transforms
	unsigned int grey, grey2;    // Used when applying greying effects


	// objectCount can be maximum 50

	switch (algo)
	{
		
		case ALGO_RAMI_LEVI:
		{
			if (frameNum == 0) {
				g_tracker.init(width, height, image_size, 0.5);
			}


			g_tracker.process((void*)pData, pObjects);
			*objectCount = pObjects->reserved1_personsCount;
			//g_tracker.draw();


			frameNum++;

		}
		break;
		case ALGO_ZOSMAN:
		{
			// benny put your algo of miltery helemet detection here
		}
		break;

		// Zero out the green and blue components to leave only the red
		// so acting as a filter - for better visual results, compute a
		// greyscale value for the pixel and make that the red component

	case ALGO_RED:

		prgb = (RGBTRIPLE*)pData;
		for (iPixel = 0; iPixel < image_size; iPixel++, prgb++) {
			prgb->rgbtGreen = 0;
			prgb->rgbtBlue = 0;
		}
		break;

	case ALGO_GREEN:

		prgb = (RGBTRIPLE*)pData;
		for (iPixel = 0; iPixel < image_size; iPixel++, prgb++) {
			prgb->rgbtRed = 0;
			prgb->rgbtBlue = 0;
		}
		break;

	case ALGO_BLUE:
		prgb = (RGBTRIPLE*)pData;
		for (iPixel = 0; iPixel < image_size; iPixel++, prgb++) {
			prgb->rgbtRed = 0;
			prgb->rgbtGreen = 0;
		}
		break;

		// Bitwise shift each component to the right by 1
		// this results in the image getting much darker

	case ALGO_DARKEN:

		prgb = (RGBTRIPLE*)pData;
		for (iPixel = 0; iPixel < image_size; iPixel++, prgb++) {
			prgb->rgbtRed = (BYTE)(prgb->rgbtRed >> 1);
			prgb->rgbtGreen = (BYTE)(prgb->rgbtGreen >> 1);
			prgb->rgbtBlue = (BYTE)(prgb->rgbtBlue >> 1);
		}
		break;

		// Toggle each bit - this gives a sort of X-ray effect

	case ALGO_XOR:
		prgb = (RGBTRIPLE*)pData;
		for (iPixel = 0; iPixel < image_size; iPixel++, prgb++) {
			prgb->rgbtRed = (BYTE)(prgb->rgbtRed ^ 0xff);
			prgb->rgbtGreen = (BYTE)(prgb->rgbtGreen ^ 0xff);
			prgb->rgbtBlue = (BYTE)(prgb->rgbtBlue ^ 0xff);
		}
		break;

		// Zero out the five LSB per each component

	case ALGO_POSTERIZE:
		prgb = (RGBTRIPLE*)pData;
		for (iPixel = 0; iPixel < image_size; iPixel++, prgb++) {
			prgb->rgbtRed = (BYTE)(prgb->rgbtRed & 0xe0);
			prgb->rgbtGreen = (BYTE)(prgb->rgbtGreen & 0xe0);
			prgb->rgbtBlue = (BYTE)(prgb->rgbtBlue & 0xe0);
		}
		break;

		// Take pixel and its neighbor two pixels to the right and average
		// then out - this blurs them and produces a subtle motion effect

	case ALGO_BLUR:
		prgb = (RGBTRIPLE*)pData;
		for (y = 0; y < height; y++)
		{
			for (x = 2; x < width; x++, prgb++) {
				prgb->rgbtRed = (BYTE)((prgb->rgbtRed + prgb[2].rgbtRed) >> 1);
				prgb->rgbtGreen = (BYTE)((prgb->rgbtGreen + prgb[2].rgbtGreen) >> 1);
				prgb->rgbtBlue = (BYTE)((prgb->rgbtBlue + prgb[2].rgbtBlue) >> 1);
			}
			prgb += 2;
		}
		break;

		// An excellent greyscale calculation is:
		//      grey = (30 * red + 59 * green + 11 * blue) / 100
		// This is a bit too slow so a faster calculation is:
		//      grey = (red + green) / 2

		case ALGO_GREY:
		{
			prgb = (RGBTRIPLE*)pData;
			for (iPixel = 0; iPixel < image_size; iPixel++, prgb++) {
				grey = (prgb->rgbtRed + prgb->rgbtGreen) >> 1;
				prgb->rgbtRed = prgb->rgbtGreen = prgb->rgbtBlue = (BYTE)grey;
			}
		}
		break;

		// Really sleazy emboss - rather than using a nice 3x3 convulution
		// matrix, we compare the greyscale values of two neighbours. If
		// they are not different, then a mid grey (128, 128, 128) is
		// supplied.  Large differences get father away from the mid grey

	case ALGO_EMBOSS:
		{
			prgb = (RGBTRIPLE*)pData;
			for (y = 0; y < height; y++)
			{
				grey2 = (prgb->rgbtRed + prgb->rgbtGreen) >> 1;
				prgb->rgbtRed = prgb->rgbtGreen = prgb->rgbtBlue = (BYTE)128;
				prgb++;

				for (x = 1; x < width; x++) {
					grey = (prgb->rgbtRed + prgb->rgbtGreen) >> 1;
					temp = grey - grey2;
					if (temp > 127) temp = 127;
					if (temp < -127) temp = -127;
					temp += 128;
					prgb->rgbtRed = prgb->rgbtGreen = prgb->rgbtBlue = (BYTE)temp;
					grey2 = grey;
					prgb++;
				}
			}
		}
		break;
	}
	return 1;
}

 

 