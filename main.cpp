#include "segment.hpp"
#include <opencv2/core/utility.hpp>
// #include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>



using namespace std;
using namespace cv;




int main(int argc, const char** argv) 
{

    std::string video = argv[1];
    VideoCapture cap(video,cv::CAP_FFMPEG);

    Mat frame;
    cap >> frame;
    // target bounding box
    Rect roi;
    roi = selectROI("tracker", frame, true, false);
    if (roi.width == 0 || roi.height == 0)
        return 0;


    return 0;
}

