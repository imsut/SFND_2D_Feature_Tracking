# Camera Based 2D Feature Tracking

## MP.1 Data Buffer Optimization

The implmeentation was changed so that, when a new image is loaded, the image is pushed to the end of the vector, then the first element is erased until the vector length becomes 2.
This is still inefficient because when you erase an element at the head, `std::vector::erase`'s completixy is `O(n)` where `n` is the length of the vector.
Although this is not a huge problem in this exercise as `n == 2`, more an efficient implementation would be to use `std::list`.


## MP.2 Keypoint Detection

"HARRIS" is implemented with `cv::cornerHarris`.
The other modern detectors are constructed with their corresponding factory methods (e.g. `cv::FastFeatureDetector::create()` for "FAST") once before the main loop.
The constructed instance is used for all ten images, which should reduce latency (but did not measure).


## MP.3 Keypoint Removal

By using `cv::Rect::contains` method, eacy keypoint is checked whether or not it's on the preceding vehicle. The resulting vector of keypoints only contains the ones on the preceding vehicle.

## MP.4 Keypoint Descriptors

Descriptor extractors are constructed with their corresponding factory methods (e.g. `cv::xfeatures2d::BriefDescriptorExtractor::create()` for "BRIEF") once before the main loop.
The constructed instance is used for all ten images, which reduces latency by 100~200 ms per image.

## MP.5 Descriptor Matching
## MP.6 Descriptor Distance Ratio

`cv::FlannBasedMatcher::create()` is used for FLANN matcher.
For k-nearest neighbor selection, `knnMatch` method on the matcher is used to extract two best matches per feature, then each pair is checked if the top one is much better than the second one.


## MP.7 Performance Evaluation 1

|                            | SHITOMASI | HARRIS | FAST | BRISK    | ORB      | AKAZE         | SIFT     | 
|----------------------------|-----------|--------|------|----------|----------|---------------|----------| 
| #Keypoints in Image 0      | 125       | 8      | 1347 | 264      | 92       | 166           | 138      | 
| #Keypoints in Image 1      | 118       | 8      | 1398 | 282      | 102      | 157           | 132      | 
| #Keypoints in Image 2      | 123       | 8      | 1375 | 282      | 106      | 161           | 124      | 
| #Keypoints in Image 3      | 120       | 8      | 1352 | 277      | 113      | 155           | 137      | 
| #Keypoints in Image 4      | 120       | 7      | 1362 | 297      | 109      | 163           | 134      | 
| #Keypoints in Image 5      | 113       | 7      | 1424 | 279      | 125      | 164           | 140      | 
| #Keypoints in Image 6      | 114       | 9      | 1382 | 289      | 130      | 173           | 137      | 
| #Keypoints in Image 7      | 123       | 9      | 1342 | 272      | 129      | 175           | 148      | 
| #Keypoints in Image 8      | 111       | 12     | 1386 | 266      | 127      | 177           | 159      | 
| #Keypoints in Image 9      | 112       | 15     | 1358 | 254      | 128      | 179           | 137      | 
| Keypoint size distribution | 4         | 10     | 7    | 8.4 - 72 | 31 - 111 | 4.8 - 22.8328 | 1.8 - 42 | 


## MP.8 Performance Evaluation 2
## MP.9 Performance Evaluation 3

The results are attached below. Based on the latency and the number of matches, the top 3 detector/descriptor combinations are

1. FAST / BRIEF (4.72 ms latency with 354 matches per image)
2. FAST / ORB   (7.31 ms latency with 345 matches per image)
3. FAST / BRISK (7.62 ms latency with 273 matches per image)

FAST detector finds much more keypoints than other detectors for this series of images with low latency.


Note 1. AKAZE descriptor can be used only with AKAZE detector. See [OpenCV API reference](https://docs.opencv.org/3.4/d8/d30/classcv_1_1AKAZE.html#details).
Note 2. SIFT / ORB combination was not able to run for insufficient memory on my environment.


| Detector / Descriptor | Detector latency | Descriptorl latency | Total latency | #matches | 
|-----------------------|------------------|---------------------|---------------|----------| 
| SHITOMASI / BRISK     | 19.3781          | 2.27063             | 21.64873      | 95.875   | 
| SHITOMASI / BRIEF     | 19.2175          | 1.51352             | 20.73102      | 118      | 
| SHITOMASI / ORB       | 19.0261          | 4.205               | 23.2311       | 113.375  | 
| SHITOMASI / FREAK     | 19.8432          | 6.24124             | 26.08444      | 96       | 
| SHITOMASI / SIFT      | 17.6768          | 20.428              | 38.1048       | 117      | 
| HARRIS / BRISK        | 20.7196          | 1.05446             | 21.77406      | 8.375    | 
| HARRIS / BRIEF        | 20.6236          | 1.01127             | 21.63487      | 9        | 
| HARRIS / ORB          | 22.9115          | 4.08919             | 27.00069      | 9        | 
| HARRIS / FREAK        | 19.9234          | 5.28248             | 25.20588      | 8.625    | 
| HARRIS / SIFT         | 17.1614          | 18.5276             | 35.689        | 8.5      | 
| FAST / BRISK          | 2.64325          | 4.97614             | 7.61939       | 272.875  | 
| FAST / BRIEF          | 2.78689          | 1.93397             | 4.72086       | 353.875  | 
| FAST / ORB            | 2.70873          | 4.60498             | 7.31371       | 345.25   | 
| FAST / FREAK          | 2.8374           | 8.14387             | 10.98127      | 279.125  | 
| FAST / SIFT           | 2.71156          | 29.2907             | 32.00226      | 365.5    | 
| BRISK / BRISK         | 49.2329          | 3.7677              | 53.0006       | 196.25   | 
| BRISK / BRIEF         | 49.372           | 1.0683              | 50.4403       | 213      | 
| BRISK / ORB           | 48.9989          | 15.0158             | 64.0147       | 188.75   | 
| BRISK / FREAK         | 48.1655          | 6.40846             | 54.57396      | 190.5    | 
| BRISK / SIFT          | 50.0631          | 34.8519             | 84.915        | 211.5    | 
| ORB / BRISK           | 9.40834          | 1.73757             | 11.14591      | 93.875   | 
| ORB / BRIEF           | 9.58172          | 0.75882             | 10.34054      | 68.125   | 
| ORB / ORB             | 8.67358          | 16.3209             | 24.99448      | 95.125   | 
| ORB / FREAK           | 9.19391          | 5.98602             | 15.17993      | 52.5     | 
| ORB / SIFT            | 9.84673          | 43.7083             | 53.55503      | 97.125   | 
| AKAZE / BRISK         | 89.5171          | 2.47767             | 91.99477      | 151.875  | 
| AKAZE / BRIEF         | 91.1481          | 1.10057             | 92.24867      | 158.25   | 
| AKAZE / ORB           | 94.3068          | 11.9926             | 106.2994      | 148.25   | 
| AKAZE / FREAK         | 92.3774          | 6.00889             | 98.38629      | 148.375  | 
| AKAZE / AKAZE         | 92.1834          | 79.6777             | 171.8611      | 157.375  | 
| AKAZE / SIFT          | 86.3634          | 24.9353             | 111.2987      | 162.125  | 
| SIFT / BRISK          | 130.442          | 2.6847              | 133.1267      | 74       | 
| SIFT / BRIEF          | 135.362          | 1.21881             | 136.58081     | 87.75    | 
| SIFT / FREAK          | 151.09           | 6.78427             | 157.87427     | 74.125   | 
| SIFT / SIFT           | 123.6            | 100.489             | 224.089       | 102.25   | 
