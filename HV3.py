import cv2
import numpy as np
import utils

cap = cv2.VideoCapture('./A4-target-2.MOV')
webcam = True
path = './image.png'

# is given (camera parametr)
scale = 3

# A4 linear dimensions (millimeter)
wP = 210*scale
hP = 297*scale


while(cap.isOpened()):
    if webcam:
        ret, frame = cap.read()
        dst = np.ndarray(shape=frame.shape,dtype=frame.dtype)
    else:
        frame = cv2.imread(path)
        ret = True
        dst = np.ndarray(shape=frame.shape,dtype=frame.dtype)

    # for ease perception (effects the result, need an amendment factor, in K matrix)
    frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)



    if ret == True:

        cv2.imshow('Camera', frame)
        imgFrame = frame.copy()
        h10 , w10 = frame.shape[:2]

        img, conts = utils.Contours(frame, showCanny=False,
                                             quality=False,
                                             minArea = 50000, filter = 4,
                                             draw=True)

        K = np.float64([[wP, 0, 0.5*(wP-1)],
                        [0, wP, 0.5*(hP-1)],
                        [0, 0, 1.0]])


        # Distortion coefs, not used
        #dist_coef = np.zeros(4)


        if len(conts) != 0:
            biggest = conts[0][2]

            #This A4 projection perspective
            imgWarp2, matrixH = utils.warpImgH (img, biggest, wP, hP)
            _,Rs,Ts,Ns = cv2.decomposeHomographyMat(matrixH,K)
            print ('===================================')
            print ('         HOMOGRAPHY MATRIX')
            print ('===================================')
            print ('\n  {0} \n'.format(matrixH))
            print ('===================================')
            print ('    Rotation solutions:\n {0}'.format(Rs))
            print ('    Translation solutions:\n {0}'.format(Ts))
            print ('=========== Found solutions of Rotation vector [YXZ] (Euler XYZ) ============')
            v1 = utils.rot_matrix_to_euler(Rs[0])
            v2 = utils.rot_matrix_to_euler(Rs[1])
            v3 = utils.rot_matrix_to_euler(Rs[2])
            v4 = utils.rot_matrix_to_euler(Rs[3])
            print ('Rotation vector1:\n {0}'.format(v1))
            print ('Rotation vector2:\n {0}'.format(v2))
            print ('Rotation vector3:\n {0}'.format(v3))
            print ('Rotation vector4:\n {0}'.format(v4))
            print ('=========== Found solutions of Translation vector [YXZ] (Meters XYZ) ============')
            t1 = utils.trans_matrix_to_vector(Ts[0])
            t2 = utils.trans_matrix_to_vector(Ts[1])
            t3 = utils.trans_matrix_to_vector(Ts[2])
            t4 = utils.trans_matrix_to_vector(Ts[3])
            print ('Translation vector1:\n {0}'.format(t1))
            print ('Length :\n {0}'.format(t1[3]))
            print ('Translation vector2:\n {0}'.format(t2))
            print ('Length :\n {0}'.format(t2[3]))
            print ('Translation vector3:\n {0}'.format(t3))
            print ('Length :\n {0}'.format(t3[3]))
            print ('Translation vector4:\n {0}'.format(t4))
            print ('Length :\n {0}'.format(t4[3]))

            h = hP
            w = wP
            New_Mask_Frame = utils.giveMask (img, conts)
            Mask_Frame_C, Point_C = utils.darwCentrCoutures(New_Mask_Frame,img, conts)
            cv2.imshow('Mask', New_Mask_Frame)
            cv2.imshow('Rectified perspective', imgWarp2)
            cv2.imshow('Frame with data on frame', img)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()
