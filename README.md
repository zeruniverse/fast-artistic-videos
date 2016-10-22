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
We just need optical flow to keep temporal consistency, not necessary `deepflow`. But we only tested with `deepflow`.    
+ To run parallely as *k* processes:    
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* 1 k`   
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* 2 k`   
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* 3 k`   
  + ...      
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* k-1 k`  
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* k k`   
