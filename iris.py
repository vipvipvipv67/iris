#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import pickle
import glob 
import pandas as pd
from scipy.io import loadmat


# In[4]:



def process_iris_image(img):
    pupil_circle = detect_pupil(img)
    if pupil_circle is None:
        print("Pupil not detected, skipping image.")
        return  # Skip this image as we cannot find the pupil

    iris_circle = detect_iris_based_on_pupil(img, pupil_circle)
    if iris_circle is None:
        print("Iris not detected, skipping image.")
        return  # Skip this image as we cannot find the iris

    masked_img = mask_outside_circles(img, pupil_circle, iris_circle)
    normalized_img = IrisNormalization(masked_img, pupil_circle, iris_circle)
    enhanced_img = iris_enhancement(normalized_img)
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    plt.show()

    
    
def IrisNormalization(image, inner_circle, outer_circle):
    if inner_circle is None or outer_circle is None:
        err_msg = "Invalid "
        if inner_circle is None:
            err_msg += "inner "
        if outer_circle is None:
            err_msg += "outer "
        err_msg += "circle parameters"
        raise ValueError(err_msg)  # Raise an exception with a specific error message
    localized_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    row = 64
    col = 512
    normalized_iris = np.zeros((row, col), dtype=np.uint8)

    angle = 2.0 * math.pi / col

    for j in range(col):
        inner_boundary_x = inner_circle[0] + inner_circle[2] * math.cos(angle * j)
        inner_boundary_y = inner_circle[1] + inner_circle[2] * math.sin(angle * j)

        outer_boundary_x = outer_circle[0] + outer_circle[2] * math.cos(angle * j)
        outer_boundary_y = outer_circle[1] + outer_circle[2] * math.sin(angle * j)

        for i in range(row):
            i_normalized = min(max(int(inner_boundary_y + (outer_boundary_y - inner_boundary_y) * (i / row)), 0), localized_img.shape[0] - 1)
            j_normalized = min(max(int(inner_boundary_x + (outer_boundary_x - inner_boundary_x) * (i / row)), 0), localized_img.shape[1] - 1)
            normalized_iris[i, j] = localized_img[i_normalized, j_normalized]

    res_image = 255 - normalized_iris
    return res_image

def iris_enhancement(img, illumination=False):  
    if illumination is True: # apply illumination adjustment
        #@the illumination estimation does not work well, so although I have coded them, I decided not to use them   
        # calculate the 4x32 mean_pool_transformed matrix
        mean_pool = block_reduce(img, (16,16),np.mean)

        # estimate the illumination by bicubic interpolation
        estimated_illumination = cv2.resize(mean_pool, (512, 64), interpolation =cv2.INTER_CUBIC)

        # subtract the estimated illumination from the original image. If we get negative value then set to 0
        enhanced_image = img - estimated_illumination
        enhanced_image = enhanced_image - np.amin(enhanced_image.ravel())  # rescale back to (0-255)
    
    elif illumination is False: # does not apply illumination adjustment
        enhanced_image = img - 0
        
    # perform the histogram equalization in each 32x32 region
    for row_index in range(0, enhanced_image.shape[0], 32):
        for col_index in range(0, enhanced_image.shape[1],32):
            sub_matrix = enhanced_image[row_index:row_index+32, col_index:col_index+32]
            # apply histogram equalization in each 32x32 sub block
            enhanced_image[row_index:row_index+32, col_index:col_index+32] = cv2.equalizeHist(sub_matrix.astype("uint8"))  
            
    return enhanced_image


# In[5]:


def detect_pupil(img):
    blur = cv2.GaussianBlur(img, (3, 3), 5)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(binary, kernel)
    dst = cv2.dilate(erosion, kernel)

    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=10, minRadius=0, maxRadius=200)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = sorted(circles[0], key=lambda c: -c[2])[0]  # Get the largest circle assuming it's the pupil
        # Draw the circles for visualization (you can comment these out in production)
        cv2.circle(img, (best_circle[0], best_circle[1]), best_circle[2], (0, 255, 255), 2)
        cv2.circle(img, (best_circle[0], best_circle[1]), 2, (0, 255, 255), -1)
        return best_circle
    else:
        print("Pupil not detected.")
        return None

    

def detect_iris_based_on_pupil(img, pupil_circle):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
    
    iris_circle = None
    
    if pupil_circle is not None:
        minRadius = int(pupil_circle[2] * 1.15)
        maxRadius = int(pupil_circle[2] * 3.25)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=150, param2=55, minRadius=minRadius, maxRadius=maxRadius)
        
        best_circle = None
        best_distance = float('inf')
        
        if circles is not None:
            for circle in circles[0]:
                distance = np.linalg.norm(np.array([circle[0]-pupil_circle[0], circle[1]-pupil_circle[1]]))
                if distance < best_distance and minRadius < circle[2] < maxRadius:
                    best_distance = distance
                    best_circle = circle

            if best_circle is not None:
                iris_circle = best_circle
                center_x = int(best_circle[0])
                center_y = int(best_circle[1])
                radius = int(best_circle[2])
                cv2.circle(img, (center_x, center_y), radius, (255, 0, 0), 2)
                cv2.circle(img, (center_x, center_y), 2, (255, 0, 0), -1)
    return iris_circle


def mask_outside_circles(img, pupil_circle, iris_circle):
    mask = np.zeros_like(img)
    if pupil_circle is not None and iris_circle is not None:
        # Convert circle centers to integers
        pupil_center = (int(pupil_circle[0]), int(pupil_circle[1]))
        iris_center = (int(iris_circle[0]), int(iris_circle[1]))
        
        # Draw the white circle of the iris on the black mask
        cv2.circle(mask, iris_center, int(iris_circle[2]), (255, 255, 255), -1)
        # Draw the black circle of the pupil on the white iris mask
        cv2.circle(mask, pupil_center, int(pupil_circle[2]), (0, 0, 0), -1)

        # Perform the bitwise operation with the original image
        result = cv2.bitwise_and(img, mask)
        return result
    return img


# In[6]:


def process_all_iris_images(base_path):
    for person_folder in os.listdir(base_path):
        person_path = os.path.join(base_path, person_folder)
        if os.path.isdir(person_path):
            inner_folder_path = os.path.join(person_path, "1")
            if os.path.exists(inner_folder_path):
                for image_name in os.listdir(inner_folder_path):
                    if image_name.endswith('.bmp'):
                        image_path = os.path.join(inner_folder_path, image_name)
                        image = cv2.imread(image_path)
                        process_iris_image(image)

if __name__=="__main__":
    process_all_iris_images("CASIA Iris Image Database (version 1.0)")


# In[7]:


from scipy import ndimage
import math
import numpy as np

###this is feature extraction and works
#defined filter
def defined_filter(x,y,f):
    m1= math.cos(2*math.pi*f*math.sqrt(x**2*y**2))
    return m1

space_constant_x1=3
space_constant_x2=4.5
space_constant_y=1.5


f1=0.1
f2=0.07
x1=range(-9,10,1)
x2=range(-14,15,1)
y=range(-5,6,1)


def gabor_filter(x,y,space_constant_x,space_constant_y,f):
    m1=defined_filter(x,y,f)
    return (1/(2*math.pi*space_constant_x*space_constant_y)
            *np.exp(-1/2*(x**2/(space_constant_x**2)+y**2/(space_constant_y**2))))*m1




def FeatureExtraction(roi):
    filter1=[]
    filter2=[]
    f1=0.1
    f2=0.07
    x1=range(-9,10,1)
    x2=range(-14,15,1)
    y=range(-5,6,1)
    space_constant_x1=3
    space_constant_x2=4.5
    space_constant_y=1.5
    for j in range(len(y)):
        for i in range(len(x1)):
            cell_1=gabor_filter(x1[i], y[j], space_constant_x1, space_constant_y, f1)
            filter1.append(cell_1)
        for k in range(len(x2)):
            cell_2=gabor_filter(x2[k], y[j], space_constant_x2, space_constant_y, f2)
            filter2.append(cell_2)
    filter1=np.reshape(filter1, (len(y), len(x1)))
    filter2=np.reshape(filter2, (len(y), len(x2)))
    
    filtered_eye1=ndimage.convolve(roi, np.real(filter1), mode='wrap', cval=0)
    filtered_eye2= ndimage.convolve(roi, np.real(filter2), mode='wrap', cval=0)
    
    vector=[]
    i=0
    while i<roi.shape[0]:
        j=0
        while j<roi.shape[1]:
            mean1=filtered_eye1[i:i+8,j:j+8].mean()
            mean2=filtered_eye2[i:i+8,j:j+8].mean()
            AAD1=abs(filtered_eye1[i:i+8,j:j+8]-mean1).mean()
            AAD2=abs(filtered_eye2[i:i+8,j:j+8]-mean2).mean()
            
            
            vector.append(mean1)
            vector.append(AAD1)
            vector.append(mean2)
            vector.append(AAD2)
            j=j+8
        i=i+8
    vector=np.array(vector)
    return vector

 


# In[8]:



def process_all_iris_images(base_path, train_features_path):
    feature_vectors = {}

    for person_id in range(1, 109):  # 从001到108
        person_folder = str(person_id).zfill(3)  # 保证有三位数
        person_path = os.path.join(base_path, person_folder)

        # 使用1文件夹的图像作为训练数据
        train_session_path = os.path.join(person_path, "1")
        if os.path.exists(train_session_path):
            for image_name in os.listdir(train_session_path):
                if image_name.endswith('.bmp'):
                    image_path = os.path.join(train_session_path, image_name)
                    image = cv2.imread(image_path, 0)  # 假设图像是灰度的
                    # 确保ROI是48x512
                    roi = cv2.resize(image, (512, 48))
                    vector = FeatureExtraction(roi)
                    feature_vectors[f"{person_folder}_1_{image_name}"] = vector

    # 保存训练数据的特征向量
    np.save(train_features_path, feature_vectors)



# In[9]:


if __name__ == "__main__":
    train_features_file = "train_features.npy"
    process_all_iris_images("CASIA Iris Image Database (version 1.0)", train_features_file)



# In[10]:


data = np.load("train_features.npy", allow_pickle=True).item()

# 查看数据
for key, value in data.items():
    print(f"Image: {key} --> Feature Vector Size: {value.shape}")



# In[11]:


import sklearn
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
# import matplotlib.pyplot as plt

def IrisMatching(lda, trainset, result, f, dis_function=1):
	min_dist = 10000000.0
	res = 0
	# f = lda.transform([target])[0]
	for i in range(len(result)):
		fi = trainset[i]
		dist = 0
		if dis_function == 1:
			dist = np.sum(np.abs(np.array(f)-np.array(fi)))
		elif dis_function == 2:
			dist = np.linalg.norm(np.array(f)-np.array(fi))
		elif dis_function == 3:
			dist = 1-(np.dot(fi, f)/(np.linalg.norm(fi)*np.linalg.norm(f)))
		if dist < min_dist:
			min_dist = dist
			res = result[i]
	return res

def IrisMatchingDist(lda, trainset, result, f, dis_function=1):
	min_dist = 10000000.0
	# f = lda.transform([target])[0]
	for i in range(len(result)):
		fi = trainset[i]
		dist = 0
		if dis_function == 1:
			dist = np.sum(np.abs(np.array(f)-np.array(fi)))
		elif dis_function == 2:
			dist = np.linalg.norm(np.array(f)-np.array(fi))
		elif dis_function == 3:
			dist = 1-(np.dot(fi, f)/(np.linalg.norm(fi)*np.linalg.norm(f)))
		if dist < min_dist:
			min_dist = dist
	return min_dist



# In[12]:


import cv2
import numpy as np
from scipy import ndimage


if __name__ == "__main__":
    dir_path = './CASIA Iris Image Database (version 1.0)/'
    X = []
    y = []

    for p in range(1, 109):
        for s in range(1, 5):
            image_dir = f"{dir_path}{p:03d}/2/{p:03d}_2_{s}.bmp"
            print(image_dir)
            image = cv2.imread(image_dir, 0)
            if image is not None:
                fea = FeatureExtraction(image)  # 确保传递正确的变量，这里是 image，不是 roi
                X.append(fea)
                y.append(p)
                
                with open('test48.txt', 'a') as f:
                    result = ' '.join(map(str, fea))
                    f.write(f"{result} {p}\n")
            else:
                print(f"Image at {image_dir} could not be loaded.")


# In[13]:


import random
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
# 假设IrisMatching和IrisMatchingDist是已经正确实现的

# ...（其他函数保持不变，除非需要调整）

def get_one_round_tmp(train, result_set):
	res = []
	res_result = []
	for i in range(0, 108):
		t = random.randint(0,len(train)-1)
		res.append(train[t])
		res_result.append(result_set[t])
	return res, res_result

def get_one_round_test(test, test_result, result_set):
	res = []
	res_result = []
	for i in range(0, len(result_set)):
		m = int(result_set[i])
		t = random.randint(0,2)
		res.append(test[(m-1)*3+t])
		res_result.append(test_result[(m-1)*3+t])
	return res, res_result


def calculate_one(tmp, tmp_result, tmp_test, tmp_test_result, measure='L1', n_components=200):
    # 初始化LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(np.array(tmp), np.array(tmp_result))
    tmp_new = lda.transform(tmp)

    # 初始化匹配和非匹配计数器
    correct_matches = 0
    total = len(tmp_test)

    # 转换测试数据
    test_new = lda.transform(tmp_test)

    for i in range(total):
        try:
            # 根据度量选择不同的相似度计算方法
            if measure == 'L1':
                dist = IrisMatchingDist(lda, tmp_new, tmp_result, test_new[i], 3, measure='L1')
            elif measure == 'L2':
                dist = IrisMatchingDist(lda, tmp_new, tmp_result, test_new[i], 3, measure='L2')
            else: # 默认使用Cosine similarity
                dist = IrisMatchingDist(lda, tmp_new, tmp_result, test_new[i], 3, measure='Cosine')
            
            # 如果匹配成功，增加计数
            if dist <= some_threshold:  # 'some_threshold' 需要定义一个合适的阈值
                correct_matches += 1
        except Exception as e:
            print(f"Error during matching: {e}")

    # 计算正确识别率（CRR）
    crr = correct_matches / total
    return crr

def calculateCRR(template, template_result, test, test_result, measure='L1', n_components=200):
    crr_accumulated = 0.0
    rounds = 500

    for i in range(rounds):
        tmp, tmp_result = get_one_round_tmp(template, template_result)
        tmp_test, tmp_test_result = get_one_round_test(test, test_result, tmp_result)
        crr_accumulated += calculate_one(tmp, tmp_result, tmp_test, tmp_test_result, measure, n_components)

    # 计算500轮的平均CRR
    average_crr = crr_accumulated / rounds
    print(f"Average CRR: {average_crr:.2f}")
    return average_crr

# 这里只展示了如何修改calculate_one和calculateCRR函数。
# 需要注意的是，这里的'some_threshold'需要根据实际情况来确定，它是决定是否匹配成功的阈值。

# 主程序部分需要调用 calculateCRR 函数，提供模板数据、结果以及n_components参数等。
# 例如：
# average_crr_L1 = calculateCRR(temple, temple_result, test, test_result, measure='L1', n_components=100)
# 然后可以为L2和Cosine相似度重复上述操作，确保每一种度量的CRR都大于等于80%。
def calculateROC(temple, tmp_result, test, test_result):
	lda = LinearDiscriminantAnalysis(n_components=100)
	lda.fit(np.array(temple),np.array(tmp_result))
	tmp_new = lda.transform(temple)
	test_new = lda.transform(test)

	roc_result = []
	tmp_score = []
	cnt = 0
	for i in range(len(test_result)):
		res = IrisMatching(lda, tmp_new, tmp_result, test_new[i], 3)
		dist = IrisMatchingDist(lda, tmp_new, tmp_result, test_new[i], 3)
		if res == test_result[i]:
			roc_result.append(1)
			tmp_score.append(dist)
		else:
			roc_result.append(0)
			tmp_score.append(dist)
	fpr,tpr,threshold = roc_curve(roc_result, tmp_score) 
	for i in range(0,len(tpr)):
		tpr[i] = 1-tpr[i]
	print (threshold)
	print (fpr)
	print (tpr)

	roc_auc = auc(tpr,fpr) 
	lw = 2  
	plt.figure(figsize=(10,10))  
	plt.plot(fpr, tpr, color='darkorange',  
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线  
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
	plt.xlim([0.0, 1.0])  
	plt.ylim([0.0, 1.05])  
	plt.xlabel('False Positive Rate')  
	plt.ylabel('True Positive Rate')  
	plt.title('Receiver operating characteristic example')  
	plt.legend(loc="lower right")  
	plt.show()  

