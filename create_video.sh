#!/bin/bash

# Set the input directory containing the PNG files
input_dir="exp/"

# Set the output video file name
output_file="output.mp4"

# Use ffmpeg to create the video
ffmpeg -framerate 30 -pattern_type glob -i "$input_dir/sanofi*.jpg" "$output_file"
