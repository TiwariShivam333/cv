#! /usr/bin/python3
# -*- encoding: utf-8 -*-

import cv2

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(color, gray):

    smiles = smile_cascade.detectMultiScale(gray, 1.85, 20)

    for (x, y, w, h) in smiles:
        cv2.rectangle(color, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return color

video = cv2.VideoCapture(0)

while True:

    _, face = video.read()

    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

    result = detect(face, gray)

    cv2.imshow('Smile detector', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
