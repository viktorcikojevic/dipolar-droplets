# !/bin/bash
ffmpeg -framerate 25 -start_number 0 -i  2D_snapshot_%d_xy.png -c:v libx264 -r 40  out.mp4

