# Parking_space_detection_and_tracking

The primary goal of this project was to detect and track empty parking spaces available in a parking lot which is a huge problem now a days in urban areas where parking is hard to find.

This approach takes information in the form of entropy and standard deviation of each selected parking spaces(to be tracked)and feeds that info to a SVM model to detect the current state of the parking lot, i.e weather any car is parked at that position or not and indicates with a bounding rectange around that parking area.

The code is divided into three parts for easy understanding:
Part1: Selecting the region of interest(ROI) manually once, for any new parking lot
Part2: Generating data to feed the SVM, Here I randomly took data form only 2-3 frames from the whole video
Part3: Testing the trained SVM model on or parking lot videos.

Note:
For any new video you have to follow all steps as the mask created in the first step will be for that specific video...Enjoy!!
