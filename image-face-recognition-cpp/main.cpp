#include <QCoreApplication>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

Mat detect(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade, Mat &colorImage, Mat &grayImage)
{
    vector<Rect> faces;

    faceCascade.detectMultiScale(grayImage, faces, 1.3, 5);

    for (Rect &face : faces) {
        rectangle(colorImage, face, Scalar(255, 0, 0), 2);

        vector<Rect> eyes;

        Mat grayFaceRoi = grayImage(face);
        Mat colorImageRoi = colorImage(face);

        eyeCascade.detectMultiScale(grayFaceRoi, eyes, 1.3, 5);

        for (Rect &eye : eyes)
            rectangle(colorImageRoi, eye, Scalar(0, 255, 0), 2);
    }

    return colorImage;
}

int main()
{
    CascadeClassifier faceCascade("/home/qt/opencv/cascades/haarcascade_frontalface_default.xml");
    CascadeClassifier eyeCascade("/home/qt/opencv/cascades/haarcascade_eye.xml");

    Mat frame = imread("/home/qt/opencv/images/denzel.jpg");

    if (!frame.data) {
        cout << "Not loaded" << endl;
        return 0;
    }

    Mat gray;

    cvtColor(frame, gray, CV_BGR2GRAY);

    Mat canvas = detect(faceCascade, eyeCascade, frame, gray);

    resize(canvas, canvas, Size(), 0.5, 0.5);

    imshow("Face detection", canvas);

    waitKey(0);

    destroyAllWindows();

    return 0;
}
