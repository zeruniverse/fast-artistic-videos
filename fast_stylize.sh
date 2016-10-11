#!/bin/bash
set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}


#Find out whether ffmpeg or avconv is installed on the system
FFMPEG=ffmpeg
command -v $FFMPEG >/dev/null 2>&1 || {
  FFMPEG=avconv
  command -v $FFMPEG >/dev/null 2>&1 || {
    echo >&2 "This script requires either ffmpeg or avconv installed.  Aborting."; exit 1;
  }
}

if [ "$#" -le 1 ]; then
   echo "Usage: ./fast_stylize <path_to_video> <path_to_style_model>"
   exit 1
fi

# Parse arguments
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
filename=${filename//[%]/x}
style_model=$2

# Create output folder
mkdir -p $filename
mkdir -p ${filename}_01


echo ""
read -p "Which backend do you want to use? \
For Nvidia GPU, use cudnn if available, otherwise nn. \
For non-Nvidia GPU, use clnn. Note: You have to have the given backend installed in order to use it. [nn] $cr > " backend
backend=${backend:-nn}

if [ "$backend" == "cudnn" ] || [ "$backend" = "nn" ] || [ "$backend" = "clnn" ]; then
  echo ""
  read -p "Please enter a resolution at which the video should be processed, \
  in the format w:h, or leave blank to use the original resolution $cr > " resolution
else
  echo "Unknown backend."
  exit 1
fi

echo ""
read -p "Enter the zero-indexed ID of the GPU to use, or -1 for CPU mode (very slow!).\
 [0] $cr > " gpu
gpu=${gpu:-0}

# Save frames of the video as individual image files
if [ -z $resolution ]; then
  $FFMPEG -i $1 ${filename}_01/frame_%06d.ppm
  resolution=default
else
  $FFMPEG -i $1 -vf scale=$resolution ${filename}_01/frame_%06d.ppm
fi

start=$(date +%s.%N)

th fast_neural_style.lua \
-model ${style_model} \
-input_dir ${filename}_01/ \
-output_dir ${filename}/ \
-gpu $gpu \
-image_size 0 \
-median_filter 0

end=$(date +%s.%N)
runtime=$(python -c "print(${end} - ${start})")
echo "Fast Stylize Runtime Was $runtime Seconds"