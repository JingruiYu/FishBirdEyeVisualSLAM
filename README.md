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