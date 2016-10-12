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
+ `bash fast_stylize.sh *example.mp4* *models/starry_night.t7*`  
+ `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480*`  
+ `bash make_video.sh *example.mp4*`  
  
\*\_\_\_\* are parts you should change accordingly.  
  
Deepmatching and deepflow is super slow. Run them parallely on CPU cluster or run their GPU versions.   
+ To run parallely as *k* processes:    
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* 1 k`   
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* 2 k`   
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* 3 k`   
  + ...      
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* k-1 k`  
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* k k`   
##To Do
+ At last step, apply a 3*3 median filter.
+ Tune style_weight (relation_weight), content_weight and temporal_weight.  
+ Solve the too dark problem  
+ Use GPU deep flow  
+ Use GPU deep matching
