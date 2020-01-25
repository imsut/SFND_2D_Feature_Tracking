/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

#include <boost/program_options.hpp>
#include <boost/histogram.hpp>
#include <boost/format.hpp>

using namespace std;

void task7(const std::string detectorType, const std::vector<cv::KeyPoint> keypoints, size_t imgIndex) {
    //auto h = boost::histogram::make_histogram(boost::histogram::axis::regular<>(10, 1.0, 50.0));
    auto h = boost::histogram::make_histogram(boost::histogram::axis::regular<float, boost::histogram::axis::transform::log>(10, 1.0, 50.0));

    float minSize = 1000000.0;
    float maxSize = 0.0;
    for (const auto& kp : keypoints) {
        h(kp.size);
        minSize = std::min(minSize, kp.size);
        maxSize = std::max(maxSize, kp.size);
    }

    cout << "Detector " << detectorType << " founds " << keypoints.size()
        << " keypoints in image " << imgIndex << " with min/max = " << minSize << "/" << maxSize << "." << std::endl;

    for (const auto& x : boost::histogram::indexed(h)) {
        std::cout << boost::format("bin %i [ %.1f, %.1f ): %i\n") % x.index() % x.bin().lower() % x.bin().upper() % *x;
    }
}

void task8(const std::string& detectorType, const std::string& descriptorType, const std::vector<cv::DMatch>& matches, size_t imgIndex) {
    cout << detectorType << "/" << descriptorType << " founds " << matches.size()
        << " matches in image " << imgIndex << "." << std::endl;


}

static void visualize(const std::string& windowName, const cv::Mat& img, const vector<cv::KeyPoint>& keypoints) {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}


/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("detector-type", po::value<std::string>()->default_value("SHITOMASI"))
        ("focus-on-vehicle", po::value<bool>()->default_value(true))
        ("max-keypoints", po::value<int>()->default_value(-1), "max number of keypoints to process. -1 sets no limit.")
        ("descriptor-type", po::value<std::string>()->default_value("BRISK"))
        ("matcher-type", po::value<std::string>()->default_value("MAT_BF"))
        ("selector-type", po::value<std::string>()->default_value("SEL_NN"))
        ("visualize-keypoints", po::value<bool>()->default_value(false))
        ("visualize-matches", po::value<bool>()->default_value(false))
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "/home/kenkawamoto/personal/udacity/SFND_2D_Feature_Tracking/";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    const string detectorType = vm["detector-type"].as<std::string>();
    const string descriptorType = vm["descriptor-type"].as<std::string>(); // BRIEF, ORB, FREAK, AKAZE, SIFT
    const string matcherType = vm["matcher-type"].as<std::string>(); // MAT_BF, MAT_FLANN
    const string selectorType = vm["selector-type"].as<std::string>();       // SEL_NN, SEL_KNN

    if (descriptorType == "AKAZE" and detectorType != "AKAZE") {
        cerr << "AKAZE descriptor can be used only with AKAZE detector." << std::endl;
        return 1;
    }

    std::cout << "Detector: " << detectorType << std::endl;
    std::cout << "Descriptor: " << descriptorType << std::endl;
    std::cout << "Matcher: " << matcherType << std::endl;
    std::cout << "Selector: " << selectorType << std::endl;

    cv::Ptr<cv::Feature2D> detector = createKeypointDetector(detectorType);
    cv::Ptr<cv::DescriptorExtractor> descExtractor = createDescriptorExtractor(descriptorType);

    /* MAIN LOOP OVER ALL IMAGES */
    int64 tickSumDet = 0;
    int64 tickSumDesc = 0;
    int totalMatches = 0;

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        cout << "=== Begin processing image " << imgIndex << " ===" << std::endl;

        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        // delete an element from the front if the vector size exceeds a specified size.
        while (dataBuffer.size() > dataBufferSize) {
            dataBuffer.erase(dataBuffer.begin());
        }

        //// EOF STUDENT ASSIGNMENT
        //cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        bool visKeypoints = vm["visualize-keypoints"].as<bool>();
        int64 ticksDet = cv::getTickCount();
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, visKeypoints);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, visKeypoints);
        }
        else
        {
            // FAST, BRISK, ORB, AKAZE, SIFT
            detKeypointsModern(keypoints, imgGray, detector);
        }
        ticksDet = cv::getTickCount() - ticksDet;
        std::cout << "Detected " << keypoints.size() << " keypoints in " << 1000.0 * ticksDet / cv::getTickFrequency() << " ms" << endl;
        tickSumDet += ticksDet;

        if (visKeypoints) {
            visualize("Results: " + detectorType, imgGray, keypoints);
        }

        //// EOF STUDENT ASSIGNMENT

        int numAllKeypoints = keypoints.size();

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = vm["focus-on-vehicle"].as<bool>();
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle) {
            std::vector<cv::KeyPoint> focused;
            std::copy_if(keypoints.begin(), keypoints.end(), std::back_inserter(focused), [&vehicleRect](const cv::KeyPoint& kp) {
                return vehicleRect.contains(kp.pt);
            });
            keypoints = focused;
        }

        //task7(detectorType, keypoints, imgIndex);

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        int maxKeypoints = vm["max-keypoints"].as<int>();
        if (maxKeypoints >= 0)
        {
            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        //cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        int64 ticksDesc = cv::getTickCount();
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descExtractor);
        ticksDesc = cv::getTickCount() - ticksDesc;
        std::cout << "Found descriptors in " << 1000.0 * ticksDesc / cv::getTickFrequency() << " ms" << endl;
        tickSumDesc += ticksDesc;


        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        //cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            task8(detectorType, descriptorType, matches, imgIndex);
            totalMatches += matches.size();

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            //cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bool visMatches = vm["visualize-matches"].as<bool>();
            if (visMatches)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

        cout << "=== End processing image " << imgIndex << " ===" << std::endl;
    } // eof loop over all images

    double detLatencyMs = (1000.0 * tickSumDet / cv::getTickFrequency()) / (imgEndIndex - imgStartIndex);
    std::cout << "Average latency for " << detectorType << " latency: " << detLatencyMs << " ms" << std::endl;

    double descLatencyMs = (1000.0 * tickSumDesc / cv::getTickFrequency()) / (imgEndIndex - imgStartIndex);
    std::cout << "Average latency for " << descriptorType << " latency: " << descLatencyMs << " ms" << std::endl;

    double avgMatches = ((double)totalMatches) / (imgEndIndex - imgStartIndex - 1);
    std::cout << "Average number of matches: " << avgMatches << std::endl;

    std::ofstream tsv;
    tsv.open("/tmp/2dfeature.csv", std::ios::out | std::ios::app);
    tsv << detectorType << "/" << descriptorType << ","
        << detLatencyMs << ","
        << descLatencyMs << ","
        << avgMatches << "\n";
    tsv.close();

    return 0;
}
