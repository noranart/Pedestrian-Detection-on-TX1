#include <bits/stdc++.h>
#include <gflags/gflags.h>
#include <sol.hpp>
#include <TH/THTensor.h>
#include "objects_detection_lib.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/gil/gil_all.hpp>

#include <sys/stat.h>

using namespace boost::filesystem;
using namespace cv;
using namespace std;

DEFINE_string(config, "", "configuration");
DEFINE_string(model, "", "model");
DEFINE_string(input, "", "input");
DEFINE_string(output, "", "output");
DEFINE_int32(realtime, 1, "realtime");
DEFINE_int32(batchSize, 8, "batch size");
DEFINE_int32(width, 640, "width");
DEFINE_int32(height, 480, "height");
DEFINE_double(threshold, 0.5, "threshold");
int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  // Initialize torch and load model
  sol::state lua;
  lua.open_libraries();

  lua.script("require 'torch'; require 'cunn'; require 'cudnn'; require 'image';");
  lua.script("torch.setdefaulttensortype('torch.FloatTensor');");
  //lua.script("cudnn.benchmark = true;");
  lua.script("cudnn.fastest = true;");

  lua.script("m = torch.load('" + FLAGS_model + "'):cuda();");
  lua.script("m:evaluate();");

  stringstream ss; ss << "input = torch.FloatTensor(" << FLAGS_batchSize << ", 3, 224, 224);";
  lua.script(ss.str());

  THFloatTensor *input = lua["input"];
  float *data = input->storage->data;

  // warm up
  auto start = chrono::high_resolution_clock::now();
  for (int i=0; i < 100; i++) {
    lua.script("result = m:forward(input:cuda()):float();");
  }

  // Initialize region proposal 
  objects_detection::init_objects_detection(FLAGS_config, false /*use_ground_plane*/, false /*use_stixels*/);

  // Set up input and start 
  VideoCapture cap;
  deque<path> files;
  map<double, string> timestamped_files;
  if (FLAGS_input == "") {
    cap.open(1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, FLAGS_width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, FLAGS_height);
  } else {
    path p(FLAGS_input);
    assert(exists(p));
    if (is_regular_file(p)) {
      files.push_back(p);
    } else {
      path log = p / "log.txt";
      if (FLAGS_realtime && exists(log)) {
        ifstream in(log.string(), fstream::in);
        int frame; double time;
        while(in >> frame >> time) {
          stringstream ss; ss << setw(4) << setfill('0') << frame+1 << ".png";
          timestamped_files[time] = FLAGS_input + "/" + ss.str();
        }
      } else {
        FLAGS_realtime = 0;
        copy(directory_iterator(p), directory_iterator(), back_inserter(files));
        sort(files.begin(), files.end());
      }
    }
  }
    
  VideoWriter outputVideo, proposalVideo;
  if (FLAGS_output == "") {
    namedWindow("Proposals");
    namedWindow("Output");
  } else {
    mkdir(FLAGS_output.c_str(), 0777);
    outputVideo.open((path(FLAGS_output) / "output.avi").string(), CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(FLAGS_width, FLAGS_height));
    proposalVideo.open((path(FLAGS_output) / "proposal.avi").string(), CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(FLAGS_width, FLAGS_height));
  }

  double cur_timestamp = 0;
  double video_timestamp = 0;
  int num_proposals = 0;
  int num_frames = 0;
  while(true) {
    path filename;
    double timestamp;
    Mat image; 
    if (FLAGS_input == "") {
      timestamp = 0;
      cap >> image;
    } else if (timestamped_files.size() > 0) {
      timestamp = cur_timestamp;
      auto it = --timestamped_files.upper_bound(timestamp);
      filename = it->second;
      cout << filename << endl;
      image = imread(filename.string(), CV_LOAD_IMAGE_COLOR);
      if (++it == timestamped_files.end()) timestamped_files.clear();
    } else {
      if (files.empty()) break;
      timestamp = timestamp + 1/30.;
      filename = files.front();
      image = imread(filename.string(), CV_LOAD_IMAGE_COLOR);
      files.pop_front(); 
    }
    if (image.rows != FLAGS_height || image.cols != FLAGS_width) {
      resize(image, image, Size(FLAGS_width, FLAGS_height));
    }

    cout << timestamp << " processing " << filename << endl;
    num_frames++;
    auto start = chrono::high_resolution_clock::now();

    objects_detection::input_image_const_view_t input_view =
      boost::gil::interleaved_view(image.cols, image.rows,
        reinterpret_cast<boost::gil::rgb8c_pixel_t*>(image.data), static_cast<size_t>(image.step));

    objects_detection::set_monocular_image(input_view);
    objects_detection::compute();

    std::vector<doppia::Detection2d> detections = objects_detection::get_detections();
    cout << detections.size() << " regions detected." << endl;
    num_proposals += detections.size();
    
    Mat proposals = image.clone();
  
    int detected = 0;
    for (int i = 0; i < detections.size(); i += FLAGS_batchSize) {
      vector<Rect> bboxes;
      for (int j = 0, e = min(FLAGS_batchSize, (int)detections.size()-i); j < e; j++) {
        auto d = detections[i+j];
        int l = max(0, (int)d.bounding_box.min_corner().x()), t = max(0, (int)d.bounding_box.min_corner().y());
        Rect r(l, t, min(FLAGS_width-1, (int)d.bounding_box.max_corner().x()) - l + 1, min(FLAGS_height-1, (int)d.bounding_box.max_corner().y()) - t + 1);
        rectangle(proposals, r, Scalar(0, 0, 255));
        bboxes.push_back(r);

        Mat patch;
        image(r).convertTo(patch, CV_32FC3, 1/255.0);
        resize(patch, patch, Size(224, 224));

        float *D = (float*)patch.data;
        int step = 3*patch.cols;
        for (int r = 0, R = patch.rows; r < R; r++) {
          for (int c = 0, C = patch.cols; c < C; c++) {
            data[j*3*224*224 +             r*224 + c] = (D[step*r + 3*c+2] - 0.485) / 0.229;
            data[j*3*224*224 +   224*224 + r*224 + c] = (D[step*r + 3*c+1] - 0.456) / 0.224;
            data[j*3*224*224 + 2*224*224 + r*224 + c] = (D[step*r + 3*c] - 0.406) / 0.225;
          }
        }
      }

      lua.script("result = m:forward(input:cuda()):float();");
      THFloatTensor *result = lua["result"];
      for (int j = 0, e = min(FLAGS_batchSize, (int)detections.size()-i); j < e; j++) {
        float neg = exp(result->storage->data[j*2]), pos = exp(result->storage->data[j*2+1]);
        float p = pos / (pos + neg);
        if (p >= FLAGS_threshold) {
          detected++;
          rectangle(image, bboxes[j], Scalar(0, 0, 255 * p));
        }
      }
    }
    cout << detected << " pedestrians detected." << endl;

    auto elapsed = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    cur_timestamp = cur_timestamp + elapsed;

    if (FLAGS_output == "") {
      imshow("Proposals", proposals);
      imshow("Output", image);
      waitKey(1);
    } else {
      do {
        outputVideo << image;
        proposalVideo << proposals;
        video_timestamp += 1/30.;        
      } while (FLAGS_realtime && video_timestamp < cur_timestamp);
    }
  }
  cout << "Done! Time/frame: " << cur_timestamp * 1000 / num_frames << "ms. #Proposals/frame: " << num_proposals * 1.0 / num_frames << endl;
  if (FLAGS_output == "") waitKey(0);
  return 0;
}
