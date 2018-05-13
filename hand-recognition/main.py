import cv2
import numpy as np

cap = cv2.VideoCapture(0)

centr = None

while True:

    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    ci = 0

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        if area > max_area:
            max_area = area
            ci = i

    cnts = contours[ci]
    hull = cv2.convexHull(cnts)
    moments = cv2.moments(cnts)

    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

    cntr = (cx, cy)

    if centr is not None:
        difx = abs(centr[0] - cntr[0])
        dify = abs(centr[1] - cntr[1])

        if difx > dify:
            if centr[0] < cntr[0]:
                print('You probably moved your hand to the right')
            elif centr[0] > cntr[0]:
                print('You probably moved your hand to the left')
        elif difx < dify:
            if centr[1] < cntr[1]:
                print('You probably moved your hand up')
            elif centr[1] > cntr[1]:
                print('You probably moved your hand down')
        else:
            print('You probably didn\'t move your hand')

    centr = cntr

    drw = np.zeros(img.shape, np.uint8)

    cv2.drawContours(drw, [cnts], 0, (0, 255, 0), 2)
    cv2.drawContours(drw, [hull], 0, (0, 0, 255), 2)
    cv2.circle(drw, cntr, 5, [255, 0, 0], 2)

    cnts = cv2.approxPolyDP(cnts, 0.01 * cv2.arcLength(cnts, True), True)
    hull = cv2.convexHull(cnts, returnPoints=False)

    defects = cv2.convexityDefects(cnts, hull)

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            dist = cv2.pointPolygonTest(cnts, cntr, True)
            cv2.circle(drw, far, 5, [0, 0, 255], -1)

    cv2.imshow('Image', drw)

    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
