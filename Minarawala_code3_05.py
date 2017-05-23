import numpy as np
import cv2 as cv
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt
from sklearn import svm
import time

# Use 'Minarawala_testvideo_03.mp4' and 'Minarawala_trainimage_04.jpg' together
# Use 'Minarawala_testvideo_01.mp4' and 'Minarawala_trainimage_02.jpg' together
Video = cv.VideoCapture('Images/Test/Minarawala_testvideo_03.mp4')
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
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



#Training:

# east parking video
"""data1 = np.load('Minarawala_data_train_input_07.npy')
original_output = np.load('Minarawala_data_train_output_08.npy')"""
#print(data1.shape)
# pplot video
data1 = np.load('Minarawala_data1_train_input_09.npy')
original_output = np.load('Minarawala_data1_train_output_10.npy')
t=time.time()
clf = svm.SVC(kernel='linear')
clf.fit(data1.T, original_output)
print(time.time()-t)



frame_count = Video.get(7)
frame_rate = Video.get(5)
print(frame_count,frame_rate)
skip = np.arange(0,frame_count,50)
for i in range(skip.shape[0]):
	Video.set(1,skip[i])

	ret,image = Video.read()
	
	if(ret):
		image = cv.resize(image,(1200,700))
		fgmask = fgbg.apply(image)
		image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
		size = image_gray.shape
		#print(size)

		#Vehicle Detection

		fgmask = cv.dilate(fgmask,kernel,iterations = 2)
		fgmask = cv.erode(fgmask,kernel,iterations=4)
		fgmask = cv.dilate(fgmask,kernel)
		im2, contours, hierarchy = cv.findContours(fgmask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		if (len(contours)>1):
                  
			area1 = np.zeros((1,len(contours)))
			for i in range(len(contours)):
				area1[0,i] = cv.contourArea(contours[i])            
                      
			for j in range(len(contours)):
				if (area1[0,j]>200):          # change this threshold value for different videos for 'Minarawala_trainvideo_01.mp4' use 200 and for 'Minarawala_testvideo_03.mp4' use 1000     
					(x, y, w, h) = cv.boundingRect(contours[j])
					cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


		#Finding Local Entropy
		img1 = entropy(image_gray, disk(3))


		#Finding the Avg Local Entropies and Standard deviation of each selected parking spot for generating data to train SVM.
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

		#Prediction:
		x = clf.predict(data.T)

		count = 0
		font = cv.FONT_HERSHEY_SIMPLEX
		for i in range(0,x.shape[0]):
			if x[i] == 1:
				cv.drawContours(image,contour,i,(0,0,255),2)
			else:
				cv.drawContours(image,contour,i,(0,255,0),2)
				count += 1
		cv.putText(image,'Available Parking:{}'.format(count),(30,30),font,1,(0,0,255),2,cv.LINE_AA)
		cv.imshow('video',image)
		k = cv.waitKey(1) & 0xff
		if k == 27:   # press 'esc'
			break
	else:
		break
Video.release()
cv.destroyAllWindows()

