#!/bin/bash

for detector in SHITOMASI HARRIS FAST BRISK ORB AKAZE SIFT; do
  for descriptor in BRISK BRIEF ORB FREAK SIFT; do
    cmd="./build/2D_feature_tracking --detector-type $detector --descriptor-type $descriptor --matcher-type MAT_BF --selector-type SEL_KNN"
    echo === $cmd ===
    $cmd
  done
done

cmd="./build/2D_feature_tracking --detector-type AKAZE --descriptor-type AKAZE --matcher-type MAT_BF --selector-type SEL_KNN"
echo === $cmd ===
$cmd
