#!/bin/bash
set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}


# Find out whether ffmpeg or avconv is installed on the system
FFMPEG=ffmpeg
command -v $FFMPEG >/dev/null 2>&1 || {
  FFMPEG=avconv
  command -v $FFMPEG >/dev/null 2>&1 || {
    echo >&2 "This script requires either ffmpeg or avconv installed.  Aborting."; exit 1;
  }
}

if [ "$#" -le 0 ]; then
   echo "Usage: ./make_video <path_to_video>"
   exit 1
fi

# Parse arguments
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
filename=${filename//[%]/x}
style_image=$2


echo ""
read -p "Which backend do you want to use? \
For Nvidia GPU, use cudnn if available, otherwise nn. \
For non-Nvidia GPU, use clnn. Note: You have to have the given backend installed in order to use it. [nn] $cr > " backend
backend=${backend:-nn}

if [ "$backend" == "cudnn" ] || [ "$backend" = "nn" ] || [ "$backend" = "clnn" ]; then
  echo ""
  read -p "Please enter a resolution at which the video should be processed. \
  This value should be exactly same as what you input to fast_stylize.sh and opt_flow.sh. \
  In the format w:h, or leave blank to use the original resolution $cr > " resolution
else
  echo "Unknown backend."
  exit 1
fi

style_weight=5e0
temporal_weight=1e3
content_weight=1.5e1

echo ""
read -p "Enter the zero-indexed ID of the GPU to use, or -1 for CPU mode (very slow!).\
 [0] $cr > " gpu
gpu=${gpu:-0}


start=$(date +%s.%N)

# Perform style transfer
th artistic_video.lua \
-content_pattern ${filename}/frame_%06d.ppm \
-flow_pattern ${filename}/flow_${resolution}/backward_[%d]_{%d}.flo \
-flowWeight_pattern ${filename}/flow_${resolution}/reliable_[%d]_{%d}.pgm \
-style_weight $style_weight \
-temporal_weight $temporal_weight \
-content_weight $content_weight \
-output_folder ${filename}/ \
-backend $backend \
-gpu $gpu \
-cudnn_autotune \
-number_format %06d \
-num_iterations 30

end=$(date +%s.%N)
runtime=$(python -c "print(${end} - ${start})")
echo "Consistency Network Runtime Was $runtime Seconds"

# Create video from output images.
$FFMPEG -i ${filename}/out-%06d.png ${filename}-stylized.$extension