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

int histogram_bins = 16;
int background_ratio = 2;
double p_b;
Histogram hist_foreground;
Histogram hist_background;


Mat bgr2hsv(const Mat &img);
void extract_histograms(const Mat &image, cv::Rect region, Histogram &hf, Histogram &hb);
Mat segment_region(const Mat &image);
Mat segment_region(
        const Mat &image,
        const Point2f &object_center,
        const Size2f &template_size,
        const Size &target_size,
        float scale_factor);


Mat get_location_prior(
        const Rect roi,
        const Size2f target_size,
        const Size img_sz);

Mat get_subwindow(
        const Mat &image,
        const Point2f center,
        const int w,
        const int h,
        Rect *valid_pixels);

double get_max(const Mat &m);

inline double kernel_epan(double x)
{
    return (x <= 1) ? (2.0/3.14)*(1-x) : 0;
}


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

    Mat image = frame(roi);

    Mat hsv_img = bgr2hsv(image);

  
    namedWindow("hsv_img",WINDOW_NORMAL);
    imshow("hsv_img",hsv_img);

    hist_foreground = Histogram(hsv_img.channels(), histogram_bins);
    hist_background = Histogram(hsv_img.channels(), histogram_bins);
    extract_histograms(hsv_img,roi,hist_foreground,hist_background);
    // Mat filter_mask;
    Point2f object_center = Point2f(static_cast<float>(image.size().width/2),
                        static_cast<float>(image.size().height/2));
    Size2f template_size = Size2f(image.size().width,image.size().height);

    Size2f original_target_size = Size2f(image.size().width,image.size().height);

    float current_scale_factor = 1;
    Mat filter_mask = segment_region(hsv_img, object_center, template_size,
                original_target_size, current_scale_factor);




    namedWindow("image",WINDOW_NORMAL);
    imshow("image",image);
    namedWindow("filter_mask",WINDOW_NORMAL);
    imshow("filter_mask",filter_mask);
    waitKey(0);


    return 0;
}

//-------------------------------------------------------------------------------------------------------
Mat bgr2hsv(const Mat &img)
{
    Mat hsv_img;
    cvtColor(img, hsv_img, COLOR_BGR2HSV);
    std::vector<Mat> hsv_img_channels;
    split(hsv_img, hsv_img_channels);
    hsv_img_channels.at(0).convertTo(hsv_img_channels.at(0), CV_8UC1, 255.0 / 180.0);
    merge(hsv_img_channels, hsv_img);
    return hsv_img;
}

void extract_histograms(const Mat &image, cv::Rect region, Histogram &hf, Histogram &hb)
{
    // get coordinates of the region
    int x1 = std::min(std::max(0, region.x), image.cols-1);
    int y1 = std::min(std::max(0, region.y), image.rows-1);
    int x2 = std::min(std::max(0, region.x + region.width), image.cols-1);
    int y2 = std::min(std::max(0, region.y + region.height), image.rows-1);

    // calculate coordinates of the background region
    int offsetX = (x2-x1+1) / background_ratio;
    int offsetY = (y2-y1+1) / background_ratio;
    int outer_y1 = std::max(0, (int)(y1-offsetY));
    int outer_y2 = std::min(image.rows, (int)(y2+offsetY+1));
    int outer_x1 = std::max(0, (int)(x1-offsetX));
    int outer_x2 = std::min(image.cols, (int)(x2+offsetX+1));

    // calculate probability for the background
    p_b = 1.0 - ((x2-x1+1) * (y2-y1+1)) /
        ((double) (outer_x2-outer_x1+1) * (outer_y2-outer_y1+1));

    // split multi-channel image into the std::vector of matrices
    std::vector<Mat> img_channels(image.channels());
    split(image, img_channels);
    for(size_t k=0; k<img_channels.size(); k++) {
        img_channels.at(k).convertTo(img_channels.at(k), CV_8UC1);
    }

    hf.extractForegroundHistogram(img_channels, Mat(), false, x1, y1, x2, y2);
    hb.extractBackGroundHistogram(img_channels, x1, y1, x2, y2,
        outer_x1, outer_y1, outer_x2, outer_y2);
    std::vector<Mat>().swap(img_channels);
}



Mat segment_region(
        const Mat &image,
        const Point2f &object_center,
        const Size2f &template_size,
        const Size &target_size,
        float scale_factor)
{
    Rect valid_pixels;
    Mat patch = get_subwindow(image, object_center, cvFloor(scale_factor * template_size.width),
        cvFloor(scale_factor * template_size.height), &valid_pixels);
    Size2f scaled_target = Size2f(target_size.width * scale_factor,
            target_size.height * scale_factor);
    Mat fg_prior = get_location_prior(
            Rect(0,0, patch.size().width, patch.size().height),
            scaled_target , patch.size());

    std::vector<Mat> img_channels;
    split(patch, img_channels);
    std::pair<Mat, Mat> probs = Segment::computePosteriors2(img_channels, 0, 0, patch.cols, patch.rows,
                    p_b, fg_prior, 1.0-fg_prior, hist_foreground, hist_background);

    Mat mask = Mat::zeros(probs.first.size(), probs.first.type());
    probs.first(valid_pixels).copyTo(mask(valid_pixels));
    double max_resp = get_max(mask);
    threshold(mask, mask, max_resp / 2.0, 1, THRESH_BINARY);
    mask.convertTo(mask, CV_32FC1, 1.0);
    return mask;
}





Mat get_subwindow(
        const Mat &image,
        const Point2f center,
        const int w,
        const int h,
        Rect *valid_pixels)
{
    int startx = cvFloor(center.x) + 1 - (cvFloor(w/2));
    int starty = cvFloor(center.y) + 1 - (cvFloor(h/2));
    Rect roi(startx, starty, w, h);
    int padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;
    if(roi.x < 0) {
        padding_left = -roi.x;
        roi.x = 0;
    }
    if(roi.y < 0) {
        padding_top = -roi.y;
        roi.y = 0;
    }
    roi.width -= padding_left;
    roi.height-= padding_top;
    if(roi.x + roi.width >= image.cols) {
        padding_right = roi.x + roi.width - image.cols;
        roi.width = image.cols - roi.x;
    }
    if(roi.y + roi.height >= image.rows) {
        padding_bottom = roi.y + roi.height - image.rows;
        roi.height = image.rows - roi.y;
    }
    Mat subwin = image(roi).clone();
    copyMakeBorder(subwin, subwin, padding_top, padding_bottom, padding_left, padding_right, BORDER_REPLICATE);

    if(valid_pixels != NULL) {
        *valid_pixels = Rect(padding_left, padding_top, roi.width, roi.height);
    }
    return subwin;
}

Mat get_location_prior(
        const Rect roi,
        const Size2f target_size,
        const Size img_sz)
{
    int x1 = cvRound(max(min(roi.x-1, img_sz.width-1) , 0));
    int y1 = cvRound(max(min(roi.y-1, img_sz.height-1) , 0));

    int x2 = cvRound(min(max(roi.width-1, 0) , img_sz.width-1));
    int y2 = cvRound(min(max(roi.height-1, 0) , img_sz.height-1));

    Size target_sz;
    target_sz.width = target_sz.height = cvFloor(min(target_size.width, target_size.height));

    double cx = x1 + (x2-x1)/2.;
    double cy = y1 + (y2-y1)/2.;
    double kernel_size_width = 1.0/(0.5*static_cast<double>(target_sz.width)*1.4142+1);
    double kernel_size_height = 1.0/(0.5*static_cast<double>(target_sz.height)*1.4142+1);

    cv::Mat kernel_weight = Mat::zeros(1 + cvFloor(y2 - y1) , 1+cvFloor(-(x1-cx) + (x2-cx)), CV_64FC1);
    for (int y = y1; y < y2+1; ++y){
        double * weightPtr = kernel_weight.ptr<double>(y);
        double tmp_y = std::pow((cy-y)*kernel_size_height, 2);
        for (int x = x1; x < x2+1; ++x){
            weightPtr[x] = kernel_epan(std::pow((cx-x)*kernel_size_width,2) + tmp_y);
        }
    }

    double max_val;
    cv::minMaxLoc(kernel_weight, NULL, &max_val, NULL, NULL);
    Mat fg_prior = kernel_weight / max_val;
    fg_prior.setTo(0.5, fg_prior < 0.5);
    fg_prior.setTo(0.9, fg_prior > 0.9);
    return fg_prior;
}

double get_max(const Mat &m)
{
    double val;
    minMaxLoc(m, NULL, &val, NULL, NULL);
    return val;
}

