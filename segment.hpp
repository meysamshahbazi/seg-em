
#ifndef SEGMENT_HPP
#define SEGMENT_HPP

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>

#include <fstream>
#include <iostream>
#include <vector>
#include <iostream>

class Histogram
{
public:
    int m_numBinsPerDim;
    int m_numDim;

    Histogram() : m_numBinsPerDim(0), m_numDim(0) {}
    Histogram(int numDimensions, int numBinsPerDimension = 8);
    void extractForegroundHistogram(std::vector<cv::Mat> & imgChannels,
            cv::Mat weights, bool useMatWeights, int x1, int y1, int x2, int y2);
    void extractBackGroundHistogram(std::vector<cv::Mat> & imgChannels,
            int x1, int y1, int x2, int y2, int outer_x1, int outer_y1,
            int outer_x2, int outer_y2);
    cv::Mat backProject(std::vector<cv::Mat> & imgChannels);
    std::vector<double> getHistogramVector();
    void setHistogramVector(double *vector);

private:
    int p_size;
    std::vector<double> p_bins;
    std::vector<int> p_dimIdCoef;

    inline double kernelProfile_Epanechnikov(double x)
        { return (x <= 1) ? (2.0/CV_PI)*(1-x) : 0; }
};


class Segment
{
public:
    static std::pair<cv::Mat, cv::Mat> computePosteriors(std::vector<cv::Mat> & imgChannels,
            int x1, int y1, int x2, int y2, cv::Mat weights, cv::Mat fgPrior,
            cv::Mat bgPrior, const Histogram &fgHistPrior, int numBinsPerChannel = 16);
    static std::pair<cv::Mat, cv::Mat> computePosteriors2(std::vector<cv::Mat> & imgChannels,
            int x1, int y1, int x2, int y2, double p_b, cv::Mat fgPrior,
            cv::Mat bgPrior, Histogram hist_target, Histogram hist_background);
    static std::pair<cv::Mat, cv::Mat> computePosteriors2(std::vector<cv::Mat> &imgChannels,
            cv::Mat fgPrior, cv::Mat bgPrior, Histogram hist_target, Histogram hist_background);

private:
    static std::pair<cv::Mat, cv::Mat> getRegularizedSegmentation(cv::Mat & prob_o,
            cv::Mat & prob_b, cv::Mat &prior_o, cv::Mat &prior_b);

    inline static double gaussian(double x2, double y2, double std2){
        return exp(-(x2 + y2)/(2*std2))/(2*CV_PI*std2);
    }
};




#endif
