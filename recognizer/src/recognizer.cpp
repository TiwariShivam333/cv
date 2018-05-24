#include "recognizer.h"

#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

Recognizer::Recognizer(const string &file)
    : m_classifier(new cv::CascadeClassifier(file))
{

}

Recognizer::~Recognizer()
{
    delete m_classifier;
}

cv::Mat Recognizer::turnGrey(cv::Mat &image)
{
    cv::Mat grey;

    cv::cvtColor(image, grey, CV_BGR2GRAY);

    return grey;
}

cv::Mat Recognizer::recognize(cv::Mat &image)
{
    cv::Mat grey = turnGrey(image);

    vector<cv::Rect> objects;

    m_classifier->detectMultiScale(grey, objects, 1.3, 5);

    draw(image, objects);

    return image;
}

void Recognizer::draw(cv::Mat &image,
                      vector<cv::Rect> &objects)
{
    for (cv::Rect obj : objects) {
        //cv::Mat roi = image(obj);

        cv::rectangle(image, obj, cv::Scalar(255, 0, 0), 3);
    }
}
