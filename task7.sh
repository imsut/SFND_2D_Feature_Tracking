$!/bin/bash

for detector in SHITOMASI HARRIS FAST BRISK ORB AKAZE SIFT; do
  ./build/2D_feature_tracking --detector-type $detector
done
