OPENCV_LIBS = `pkg-config --libs --cflags opencv`

forward: forward.cpp
	g++ -o forward -I. -I/home/ubuntu/torch/include -I/home/ubuntu/torch/include/TH -I/usr/include/eigen3 -I/home/ubuntu/Pedestrian-Detection-Project/src -std=gnu++14 -Ofast -march=native forward.cpp -lgflags -L/home/ubuntu/torch/lib -lluajit -lluaT -lboost_system -lboost_filesystem $(OPENCV_LIBS) -L. -lmonocular_objects_detection

run: forward
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):. OMP_NUM_THREADS=4 ./forward --config squarechn.ini --model resnet_18_small_fc_dropout.t7 --input $(input) --output $(output)
