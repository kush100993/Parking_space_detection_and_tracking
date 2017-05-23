import numpy as np 
import cv2 as cv

refPt = ()
finished = False
points = []
ROI = []
mask = []
bunch_4 = []
bunch_5 = []

def click(event, x, y, flags, param):
	
	global ROI,finished,mask,points
	if event == cv.EVENT_LBUTTONDOWN:
		print(x,y)
		refPt = (x, y)
		if(len(points)==0):
			image1[refPt[::-1]] = (0,0,255)
		else:
			cv.line(image1,refPt,points[len(points)-1],(0,0,255),2)
			
		points.append(refPt)
		return

	if event == cv.EVENT_RBUTTONDOWN:
		if (len(points)<4):
			print('ERROR!! there should be more than 2 points')
			return
		
		cv.line(image1,points[0],points[len(points)-1],(0,0,255),2)
		if(len(points)==4):
			bunch_4.append(points)
			points=[]
		if(len(points)==5):
			bunch_5.append(points)
			points=[]
		return
		
	if event == cv.EVENT_LBUTTONDBLCLK:
		size = image1.shape
		mask = np.zeros(size,dtype = np.uint8)

		pts1 = np.array(bunch_4,dtype = np.int32)
		print(pts1)
		pts2 = np.array(bunch_5,dtype = np.int32)		
		cv.fillPoly(mask,[pts1[k] for k in range(len(bunch_4))],(255,255,255))
		if(bunch_5 != []):
			cv.fillPoly(mask,[pts2[k] for k in range(len(bunch_5))],(255,255,255))

		mask = mask[:,:,0]/255.
		ROI = (image1[:,:,0]/255.)*mask
		finished = True
		return 
		
	

cap = cv.VideoCapture('Images/Train/Minarawala_trainvideo_03.mp4')
ret,image = cap.read()
#image=cv.imread('C:/Users/kush/Desktop/Minarawala_report_01/Minarawala_report_01_08.jpg');
size = image.shape
print(size)
image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
img = np.zeros(size,dtype = np.uint8)
img[:,:,0] = image
img[:,:,1] = image
img[:,:,2] = image

image1 = cv.resize(img,(1200,700)) 
#cv.imwrite('image1.png',image1)

cv.namedWindow("ImageDisplay",cv.WINDOW_AUTOSIZE)

cv.setMouseCallback("ImageDisplay",click,0)

#Main loop
while (finished == False):
	cv.imshow("ImageDisplay",image1);
	key = cv.waitKey(1) & 0xFF

	if key == ord("c"):
		break
   

#Change the name in imwrite to write with the name you like
#Show results
ROI1 = ROI*255
ROI1 = ROI1.astype(np.uint8)
mask1 = mask*255
mask1 = mask1.astype(np.uint8)
cv.namedWindow("Result")
cv.imshow("Result",ROI)
#cv.imwrite("Train/Minarawala_trainvideo_10.mp4",ROI1)
cv.namedWindow("Mask")
cv.imshow("Mask",mask)
cv.imwrite("Images/Train/Minarawala_trainimage_05.mp4",mask1)
cv.waitKey(0)









