#Fast Artistic Style Transfer for Video  
##How to Install
``` bash
#install torch
cd consistencyChecker
make
cd ..
bash download_models.sh
```  

##Example for Running
Below is an example to stylize video *example.mp4* (Result resolution 640:480)  
+ Put your video *example.mp4* under this folder.  
+ `bash fast_stylize.sh *example* *models/starry_night.t7*`  
+ `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480*`  
+ `bash make_video.sh *example.mp4*`  
