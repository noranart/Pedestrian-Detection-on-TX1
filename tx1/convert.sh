mkdir ~/Desktop/$1
ffmpeg -i ~/Desktop/$1.mp4 -vf scale=640:480 ~/Desktop/$1/%4d.png
ffprobe -show_frames -select_streams v ~/Desktop/$1.mp4 | grep -P "best_effort_timestamp_time|coded" | paste -d" " - - | sed 's/best_effort_timestamp_time=\(.*\) coded_picture_number=\(.*\)/\2 \1/' > ~/Desktop/$1/log.txt
cd ~/capstone
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:. OMP_NUM_THREADS=4 ./forward --config squarechn.ini --model resnet_18_small_fc_dropout.t7 --threshold 0.5 --input ~/Desktop/$1 --output ~/Desktop/$1

