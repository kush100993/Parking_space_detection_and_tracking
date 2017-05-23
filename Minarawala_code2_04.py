import numpy as np
import cv2 as cv
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt
from sklearn import svm


# Use 'Minarawala_trainvideo_03.mp4' and 'Minarawala_trainimage_04.jpg' together
# Use 'Minarawala_trainvideo_01.mp4' and 'Minarawala_trainimage_02.jpg' together
Video = cv.VideoCapture('Images/Train/Minarawala_trainvideo_03.mp4')
mask = cv.imread('Images/Train/Minarawala_trainimage_04.jpg',cv.IMREAD_GRAYSCALE)
mask1 = mask/255.
mask2 = mask1.astype(np.uint8)


#Contouring and masking of Selected parking spots
img, cnt, hei = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
area = np.zeros((1,len(cnt)))
for i in range(len(cnt)):
	area[0,i] = cv.contourArea(cnt[i])
val = area[0]>10
area = val*area[0]
idx = np.nonzero(area)
area = area[idx]
contour=[]
for i in range(area.shape[0]):
	contour.append(cnt[idx[0][i]])  #Note: idx[0] is a vector and i am using its indexing to get the contours which are greater than some area threshold.



frame_count = Video.get(7)
frame_rate = Video.get(5)
print(frame_count,frame_rate) 

#Set the frame number here in this 'Video.set' function at second position to select that particuler frame and extract training data from that frame.
Video.set(1,400)
ret,image = Video.read()
image = cv.resize(image,(1200,700))
image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
size = image_gray.shape



#Finding Local Entropy
img1 = entropy(image_gray, disk(3))



#Finding the Average Local Entropies and Standard deviations of each selected parking spot for generating data.
avg_entropies=[]
stds=[]
for i in range(0,len(contour)):
	msk = np.zeros((size))
	cv.drawContours(msk,contour,i,(1,1,1),-1)
	img2 = img1*msk
	std_val = img2[np.nonzero(img2)]
	nonzero = np.nonzero(msk)
	total = nonzero[0].shape[0]
	x = np.zeros((1,total-std_val.shape[0]))
	y = np.append(np.reshape(std_val,(1,std_val.shape[0])),x)
	std = np.std(y)
	stds.append(std)
	Sum = np.sum(img2)
	avg = Sum/total
	avg_entropies.append(avg)
		
data = np.vstack((avg_entropies,stds))
print(data) # when it prints data in the console area, copy that data to add it to our training data for SVM.	
		
cv.drawContours(image,contour,-1,(0,255,0), 2)
plt.imshow(image,cmap = plt.cm.gray)
plt.show()
Video.release()
cv.destroyAllWindows()






















