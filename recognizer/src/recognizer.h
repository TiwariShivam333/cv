#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <string>

#include <opencv2/objdetect/objdetect.hpp>

using namespace std;

class Recognizer
{
public:
    Recognizer(const string &file);
    ~Recognizer();
    cv::Mat recognize(cv::Mat &image);

private:
    cv::Mat turnGrey(cv::Mat &image);
    void draw(cv::Mat &image, vector<cv::Rect> &objects);

private:
    cv::CascadeClassifier *m_classifier;
};

#endif // RECOGNIZER_H
