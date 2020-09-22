import cv2
import numpy as np

def Contours(img, cThr=[100, 100], showCanny=False, quality=False, minArea=1000, filter=0, draw=False):
    #Light contours
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    #Hard contours
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny:
        if quality:
            cv2.imshow('Canny', imgCanny)

        else:
            cv2.imshow('Threshold', imgThre)

    if quality:
        im2, contours,hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        finalCountours = []
    else:
        im2, contours,hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        finalCountours = []
        stdCountours = []


    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append((len(approx), area, approx, bbox, i))
                    stdCountours.append(bbox)
            else:
                finalCountours.append((len(approx), area, approx, bbox, i))
                stdCountours.append(bbox)
    finalCountours = sorted(finalCountours,key = lambda x:x[1], reverse=True)
    stdCountours = sorted(finalCountours,key = lambda x:x[1], reverse=True)


    if draw:
        for con in finalCountours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)

    return img, finalCountours

def reorder(myPoints):
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    #This points defune for specific my input video flow! (0,3,1,2)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

#This points defune for specific my input video flow (horizontal)!!!! (h,w)
#If video flow was changed (vertical) then chenge order again!  (w,h)

def warpImg (img, points, h, w, pad=20):
    points = reorder (points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w,h))

    return imgWarp, matrix

def warpImgH (img, points, h, w, pad=20):

    points = reorder (points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix, maskH = cv2.findHomography(pts1,pts2,cv2.RANSAC,5)
    imgWarp = cv2.warpPerspective(img, matrix, (w,h))


    return imgWarp, matrix


def rot_matrix_to_euler(R):
    y_rot = np.arcsin(R[2][0])
    x_rot = np.arccos(R[2][2]/np.cos(y_rot))
    z_rot = np.arccos(R[0][0]/np.cos(y_rot))
    y_rot_angle = y_rot *(180/np.pi)
    x_rot_angle = x_rot *(180/np.pi)
    z_rot_angle = z_rot *(180/np.pi)
    return x_rot_angle,y_rot_angle,z_rot_angle

def trans_matrix_to_vector(R):
    y_tr = R[0]
    x_tr = R[1]
    z_tr = R[2]
    v_len = np.sqrt((R[0]*R[0]+R[1]*R[1]+R[2]*R[2]))

    return x_tr,y_tr,z_tr,v_len


#WarpPerspective CV_WARP_FILL_OUTLIERS


def giveMask (frame, contours, invent=False):

    for con in contours:

        Mask_Frame = np.zeros( (frame.shape[0],frame.shape[1]) )
        contours2 = np.array( [ [0,0], [0,frame.shape[0]], [frame.shape[0],frame.shape[1]], [frame.shape[1],0] ] )
        cv2.fillPoly(Mask_Frame,pts=[contours2],color = (0,0,0))
        pts = con[4]
        cv2.fillPoly(Mask_Frame,[np.int32(pts)],(255,255,255))
        Mask_Frame_U8 = Mask_Frame.astype('uint8')


    if invent:
        New_Mask_Frame_U8 = cv2.bitwise_not(Mask_Frame_U8)


    if invent:
        return New_Mask_Frame_U8
    else: return Mask_Frame_U8


def darwCentrCoutures (Mask_Frame_U8,imgFrame, InCntours, DrawC = True):

    im2, contoursC, hierarchy = cv2.findContours(Mask_Frame_U8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contoursC:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        PointC = (cX,cY)
        cv2.circle(imgFrame, (cX, cY), 5, (255, 255, 0), -1)
        cv2.putText(imgFrame, 'Centroid of A4 {}'.format(PointC), (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if DrawC:
        for con in InCntours:
            cv2.drawContours(imgFrame,con[4],-1,(0,255,13),3)

    return Mask_Frame_U8, PointC
