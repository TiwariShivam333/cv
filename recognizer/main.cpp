#include "recognizer.h"

#include <QCoreApplication>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    Recognizer reco("/home/aluno/recognizer/haarcascade_frontalcatface.xml");

    Mat image = imread("/home/aluno/recognizer/cat.jpg");

    Mat result = reco.recognize(image);

    imshow("Face recognizer", result);

    return a.exec();
}
