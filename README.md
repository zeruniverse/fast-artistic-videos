#Fast Artistic Style Transfer for Video   
Torch implementation of a faster video stylization approach. See details and demos in [project page](https://zeruniverse.github.io/fast-artistic-videos/).     
![demo](https://cloud.githubusercontent.com/assets/4648756/19905301/599ad5fc-a033-11e6-9956-d0898bd581d6.jpg)   
Watch demo videos on YouTube: [Demo Video](https://youtu.be/OA3AoLOyLu0), [Comparison With Other Methods](https://youtu.be/PTlaByLz6I0)    
## How to Install
``` bash
#install torch
cd consistencyChecker
make
cd ..
bash download_models.sh
```  

## Example for Running
Below is an example to stylize video *example.mp4* (Result resolution 640:480)  
+ Put your video *example.mp4* under this folder.  
+ `bash fast_stylize.sh *example.mp4* *models/starry_night.t7*`  
+ `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480*`  
+ `bash make_video.sh *example.mp4*`  
  
\*\_\_\_\* are parts you should change accordingly.   
Please refer to [this project](https://github.com/jcjohnson/fast-neural-style) if you want to train your own style model.   
  
Deepmatching and deepflow is super slow. Run them parallely on CPU cluster or run their GPU versions.     
You just need optical flow to keep temporal consistency, not necessary `deepflow`. But we only tested with `deepflow`.    
+ To run parallely on *k* machines:    
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* 1 k`   
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* 2 k`   
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* 3 k`   
  + ...      
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* k-1 k`  
  + `bash opt_flow.sh *example_01/frame_%06d.ppm* *example/flow_640:480* k k`   
  
## Why Faster
Use the real-time algorithm for stylization of each frame. So L-BFGS is only used for temporal consistency purpose. L-BFGS
takes temporal loss, perceptual loss and relation loss into account and in addition, pixel loss to avoid contrast loss.       
  
## Reference  
Implementation is based on the following projects:  

+ [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://github.com/jcjohnson/fast-neural-style)  
+ [Artistic style transfer for videos](https://github.com/manuelruder/artistic-videos)   
+ [DeepFlow and DeepMatching](http://thoth.inrialpes.fr/src/deepflow/)  
  
Their license can be found in their project repository.  
  
## License
GNU GPL 3.0 for personal or research use. *COMMERCIAL USE PROHIBITED*.