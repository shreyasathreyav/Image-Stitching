import cv2
import numpy as np
import matplotlib.pyplot as plt



def stitch_background(img1, img2, savepath=''):

    def compute_sift(img1):
    
        sift = cv2.xfeatures2d.SIFT_create()
        kp_image1, desc_image1 = sift.detectAndCompute(img1, None)
        return kp_image1, desc_image1
   
    
    def SSD(desc_image1, desc_image2):
        list1 = []
        for i in range(len(desc_image1)):
            list2 = []
            for j in range(len(desc_image2)):
                error = (desc_image1[i] - desc_image2[j])
                euclid_distace = np.matmul(error.T, error)
                list2.append([i, j, euclid_distace])
            #https://www.geeksforgeeks.org/python-sort-list-according-second-element-sublist/
            list2.sort(key=lambda x: x[2])
            min_1 = list2[0]
            min_2 = list2[1]
    
            ratio = min_1[2]/min_2[2]
            if ratio < 0.6:
                list1.append(list2[0])
    
        return list1
    
    def keypoint_value(kp_image1, kp_image2, value_set): 
        kp1 = []
        kp2 = []
        for i in range(len(value_set)):
            img1_keypoint = value_set[i][0]
            img2_keypoint = value_set[i][1]
            kp1.append(kp_image1[img1_keypoint])
            kp2.append(kp_image2[img2_keypoint])
            
        pointsA = []
        pointsB = []
        
        for eachPt in kp1:
            A = np.float32(eachPt.pt)
            pointsA.append(A)
        
        for eachPt in kp2:
            B = np.float32(eachPt.pt)
            pointsB.append(B)
        
        pointsA = np.array(pointsA)
        pointsB = np.array(pointsB)
        
        return pointsA, pointsB
    
    def compute_homography(pointsA, pointsB):
        Homography, mask = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, 5.0)
        return Homography
    
    def do_warp(Homography, img1, img2):
        result = cv2.warpPerspective(img1, Homography, ((img1.shape[1]+img2.shape[1]), (img2.shape[0]+img1.shape[0])))
        return result
    
    def reshape(result, img2):
        temp = np.zeros((result.shape))
        temp[:img2.shape[0], :img2.shape[1]] = img2
        temp = temp.astype(np.float32)
        result = result.astype(np.float32)
    
        return result, temp
    
    def stitch_blend(result,temp,alpha):

        Final = np.zeros((result.shape))
        

        for i in range(Final.shape[0]):
            for j in range(Final.shape[1]):
                for k in range(Final.shape[2]):
                    if (result[i][j][k]) == 0 and (temp[i][j][k] ==0):
                        Final[i][j][k] = 0
                    else:
                        if temp[i][j][k] - result[i][j][k] >= 0:
                            Final[i][j][k] = temp[i][j][k] 
                        else:
                            Final[i][j][k] = result[i][j][k]

        x = np.where(Final != 0)
        x1 = np.min(x[0])
        x2 = np.min(x[1])
        x3 = np.max(x[0])
        x4 = np.max(x[1])
        
        final_img = Final[x1:x3, x2:x4]
        
        return final_img   
    
    def zero_paddding(image):    

        
        height_image = image.shape[0] 
        width_image = image.shape[1]
        third_image = image.shape[2]
        
        result = np.zeros((image.shape[0]+1000, image.shape[1] + 1000, 3), dtype=np.uint8)
        
        start = 500 
        result[500:500+image.shape[0],500:500+image.shape[1]]  = image
        
        return result

    img2 = zero_paddding(img2)
    kp_image1, desc_image1 = compute_sift(img1)
    kp_image2, desc_image2 = compute_sift(img2)
    
    value_set = SSD(desc_image1, desc_image2)
    

    pointsA, pointsB = keypoint_value(kp_image1, kp_image2, value_set)
    
    Homography = compute_homography(pointsA, pointsB)
    
    result = do_warp(Homography, img1, img2)
    
    img_1, img_2 = reshape(result, img2)
    
    final_image = stitch_blend(img_1, img_2, 0.2)
    
    cv2.imwrite("task1.png", final_image)
    
    return final_image
    
    
    
    
    
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
    


