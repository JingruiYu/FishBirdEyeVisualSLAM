Using the bird view and front view for SLAM

v0.0:
the original, Only Front Mono.

v0.1:
Using the Encoder code, Get the prior T. Good Performance.

v0.2:
Lost and ReInit. No test. 

v0.3:
Using GT pose for TrackB(). Modified the extern parameters.

v0.4:
Get the bird's key points. The bird mask is changed. The re initialization may have some problem.

v0.5:
Bird Match is done. the level and the inv Mat have some problem.

v0.6:
Using prior Tbw for bird match. 
Using prior Tcw for bird match. 

v0.7:
Tcw Bird Optimized; with Filter of Prior.  
less than 0.1;

v0.8:
Using bird map points for matching, and generate more map point by keypoints match.

v0.9:
Local Map for Bird were build, which are used for search match and improve the inliers.

v1.0:
Fusion Bird with Front, The map points of bird during the initiation is built.

v1.1:
Bird matched per Frame. Birs MP is added to the KF and Map.
The local Map for bird is created, as well as the local KFs.

v1.2:
the Bird points are added to both the pose optimization per frame, localMap and localMapping.

v1.3:
give a unified flag, and test four mode of both bHaveBird and bTightCouple.

v1.4:
Bird tracking is implemented when the front is lost. Some thing cout out for debug.

v1.5:
wait for change parameters, and test the lost reInit performance.

v1.6:
Hope the bird kf can be more when lost, and the parent relationship be useful when GBA after loopclosing.

v1.7:
change the outlier of front.

v1.8:
some compare experiments have done. Fix the reInit frame in the local Mapping.

v1.9:
Bird images are saved.

v2.0:
Drawer for video and guidence.