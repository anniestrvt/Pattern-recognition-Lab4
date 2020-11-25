import pickle
import cv2
import numpy as np
from scipy.spatial import distance


loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
words = pickle.load(open("words.pickle.dat", "rb"))
img1 = cv2.imread('Photos/Warhol.jpg',0)
dali = cv2.imread('Photos/Dali_new.jpg')
def find_index(image, center):
    count = 0
    ind = 0
    dist = 0
    for i in range(len(center)):
        if(i == 0):
            count = distance.euclidean(image, center[i])
            dist = count
            #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
        if(dist < count):
            ind = i
            count = dist
    return ind

def handling_image_video(img2_0):

    img2 = cv2.cvtColor(img2_0, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    kpts2,desc2 = akaze.detectAndCompute(img2, None)
    histogram = np.zeros(len(words))
    for each_feature in desc2:
        ind = find_index(each_feature, words)
        histogram[ind] += 1
    histogram = histogram.reshape(1, -1)
    y_pred = loaded_model.predict(histogram)
    if y_pred==0:
        print('here')
        img1 = cv2.imread('Photos/Warhol.jpg',0)
        dali = cv2.imread('Photos/Dali_new.jpg')
        kpts1, desc1 = akaze.detectAndCompute(img1, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        nn_matches = matcher.knnMatch(desc1, desc2, 2)
        matched1 = []
        matched2 = []
        good_matches = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for m, n in nn_matches:
            if m.distance < nn_match_ratio * n.distance:
                good_matches.append(cv2.DMatch(len(matched1), len(matched2), m.distance))
                matched1.append(kpts1[m.queryIdx])
                matched2.append(kpts2[m.trainIdx])
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # matchesMask, # draw only inliers
                           flags=2)
        src_pts = np.float32([x.pt for x in matched1]).reshape(-1, 1, 2)
        dst_pts = np.float32([x.pt for x in matched2]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        pts = np.float32([[0, 0], [0, 1269], [930, 1269], [930,0]]).reshape(-1, 1, 2)
        img1 = cv2.polylines(img1, [np.int32(pts)], True, 255, 3, cv2.LINE_AA)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        warp_mat = cv2.getPerspectiveTransform(pts, dst)
        dali_new = cv2.warpPerspective(dali, warp_mat, (dali.shape[1], dali.shape[0]))

        # I want to put logo on top-left corner, So I create a ROI
        rows, cols, ch = dali_new.shape
        roi = img2_0[0:rows, 0:cols]
        y = 0
        x = 0
        dali_new = dali_new[y:y + roi.shape[0], x:x + roi.shape[1]]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(dali_new, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(dali_new, dali_new, mask=mask)
        # Put logo in ROI and modify the main image

        dst = cv2.add(img1_bg, img2_fg)
        img2_0[0:rows, 0:cols] = dst
        #cv2.imshow('res', img2_0)
        #cv2.waitKey()
    return img2_0


def handling_image_webcam(img2_0):

    img2 = cv2.cvtColor(img2_0, cv2.COLOR_BGR2GRAY)

    akaze = cv2.ORB_create()
    kpts2,desc2 = akaze.detectAndCompute(img2, None)
    histogram = np.zeros(len(words))
    for each_feature in desc2:
        ind = find_index(each_feature, words)
        histogram[ind] += 1
    histogram = histogram.reshape(1, -1)
    y_pred = loaded_model.predict(histogram)
    if y_pred==0:
        print('here')
        img1 = cv2.imread('Photos/Warhol.jpg',0)
        dali = cv2.imread('Photos/Dali_new.jpg')
        kpts1, desc1 = akaze.detectAndCompute(img1, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        nn_matches = matcher.knnMatch(desc1, desc2, 2)
        matched1 = []
        matched2 = []
        good_matches = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for m, n in nn_matches:
            if m.distance < nn_match_ratio * n.distance:
                good_matches.append(cv2.DMatch(len(matched1), len(matched2), m.distance))
                matched1.append(kpts1[m.queryIdx])
                matched2.append(kpts2[m.trainIdx])
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # matchesMask, # draw only inliers
                           flags=2)
        src_pts = np.float32([x.pt for x in matched1]).reshape(-1, 1, 2)
        dst_pts = np.float32([x.pt for x in matched2]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        pts = np.float32([[0, 0], [0, 1269], [930, 1269], [930,0]]).reshape(-1, 1, 2)
        img1 = cv2.polylines(img1, [np.int32(pts)], True, 255, 3, cv2.LINE_AA)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        warp_mat = cv2.getPerspectiveTransform(pts, dst)
        dali_new = cv2.warpPerspective(dali, warp_mat, (dali.shape[1], dali.shape[0]))

        # I want to put logo on top-left corner, So I create a ROI
        rows, cols, ch = dali_new.shape
        roi = img2_0[0:rows, 0:cols]
        y = 0
        x = 0
        dali_new = dali_new[y:y + roi.shape[0], x:x + roi.shape[1]]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(dali_new, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(dali_new, dali_new, mask=mask)
        # Put logo in ROI and modify the main image

        dst = cv2.add(img1_bg, img2_fg)
        img2_0[0:rows, 0:cols] = dst
        #cv2.imshow('res', img2_0)
        #cv2.waitKey()
    return img2_0
#We will work in two modes: upload video (Lab3) or webcamera (Lab1)

def video(file_name_read, file_name_write):
    cap1 = cv2.VideoCapture(file_name_read)
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    out = cv2.VideoWriter(file_name_write, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    img1 = cv2.imread('Photos/Warhol.jpg', 0)
    dali = cv2.imread('Photos/Dali_new.jpg')
    akaze = cv2.AKAZE_create()
    while cap1.isOpened():
        ret, frame = cap1.read()
        if not ret:
            break

        fr = handling_image_video(frame)

        out.write(fr)

    cap1.release()
    out.release()
    cv2.destroyAllWindows()



def webcam(path1, path2):
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path1, fourcc, 20.0, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:

            out.write(frame)
            cv2.imshow('frame', frame)
            # press 'q' to stop shooting
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    img1 = cv2.imread('Photos/Warhol.jpg', 0)
    dali = cv2.imread('Photos/Dali_new.jpg')
    cap1 = cv2.VideoCapture(path1)
    out = cv2.VideoWriter(path2, fourcc, 20.0, (frame_width, frame_height))
    while cap1.isOpened():
        ret, frame = cap1.read()
        if not ret:
            break
        fr = handling_image_webcam(frame)
        out.write(fr)
        cv2.imshow('frame', fr)

        if cv2.waitKey(1) == ord('q'):
            break
    out.release()
    cap1.release()
    cv2.destroyAllWindows()

video('video1.mp4', 'video1_changed.avi')