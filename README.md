# PIXELATE_TECHNEX2020
Winning solution of Pixelate_Technex_2020 -an image processing event ,with hardware implementation of robot for solving maze by applying Dijikstra's algorithm

# Problem Statement:
Pixelate of Technecx2020 ,expected us to develop a robot for solving a maze using the live feed from a over head webcam feed.The detailed PS:(https://drive.google.com/open?id=1Ce5YPxhNc1LwUZp86alKOLzDNbr6lFp6).

# Our Solution:
1)We used opencv and other image processing techniques for shape and color segmentation from video feed.

2)A matrix was made from each detected shapes and a unique node number was given to each of the detected shape and dijikstra's algorithm was applied to get the shortest path.

3)Each color shape was given certain priority level which was used later to avoid that shape from the path given by dijikstra.

4)Then an optimal graph/path was plotted to get the optimal trajectory ,connecting the centroids of all the shapes along the way to the destination shape.

5)Once the path was recieved we used aruco marker and PID control system to make our robot to follow the line conecting all this dots.

6)Individual PID parameters were used to make the robot to go straight and as well take sharp 90 degree turns.
# Trial Run:
![pix](https://user-images.githubusercontent.com/43948945/75632104-e5a82e00-5c1e-11ea-8231-8b5b6d8e6c4c.gif)
# Our Final Run:
https://www.youtube.com/embed/MtGuLSTmohY?start=1
# Background processing:
![Screenshot 2020-03-02 at 12 14 36 AM](https://user-images.githubusercontent.com/43948945/75631726-4cc3e380-5c1b-11ea-8dab-1d024f97ede9.png)
# Generated_Path:
![36dc8cce-3f39-48c9-a9fe-0ad09e9a5234](https://user-images.githubusercontent.com/43948945/75631751-80067280-5c1b-11ea-9b30-7adec9798d40.jpg)
# Robot Used:
![8bc10981-e114-4d2b-bdeb-da8ac0134584](https://user-images.githubusercontent.com/43948945/75630637-9c51e180-5c12-11ea-9d2e-3fe17faf9937.jpg)

