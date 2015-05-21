// C++-Includes
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* getenv */
#include <dirent.h>
#include <string>
#include <time.h>

// OpenCV-Includes
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "datasetIO.hpp"
#include "globalDescriptors/custHog.hpp"
#include "globalDescriptors/BoWDescriptor.hpp"
#include "globalDescriptors/ColorStructureDescriptor.hpp"


int main ()
{
    // load dataset
    const std::string caltechPath = datasetIO::getCaltechPath();
    DIR* const caltechTopDir = opendir(caltechPath.c_str());
    if(caltechTopDir == NULL)
    {
        std::cerr << "Error while opening caltech toplevel dir. Exiting ..." << std::endl;
        std::exit(1);
    }
    const std::vector<std::string> classNames = datasetIO::getClassNames(caltechTopDir);
    datasetIO::dataSet dataset = datasetIO::getDataSet(classNames, caltechPath);

    // init local descriptors
    cv::initModule_nonfree();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SIFT");
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");

    // init BoW
    cv::TermCriteria termcrit = cv::TermCriteria();
    int attempts = 3;
    int flags = cv::KMEANS_PP_CENTERS;
    cv::Ptr<BoWDescriptor> bowDescriptor(new BoWDescriptor(extractor,matcher,400,termcrit,attempts,flags));
    cv::Ptr<GlobalDescriptor> descriptor( bowDescriptor );

    // get training images and train BoW
//    std::vector<cv::Mat> trainingImgs;
//    std::vector<size_t> classIdxs;
//    for(size_t i = 0; i < classNames.size(); ++i)
//    {
////        classIdxs.push_back(std::rand() % classNames.size());
//        std::vector<cv::Mat> classImgs = dataset.getRandomImagesFromClass(15,classNames[i], rand());
//        trainingImgs.insert(trainingImgs.end(), classImgs.begin(), classImgs.end());
//    }
//    std::cout << "training BoW.." << std::endl;
//    bowDescriptor->train(trainingImgs, true);

    // read trained vocabulary from disk
    bowDescriptor->readVocabularyFromDisk("etc/vocabulary_all_classes_15.xml");

    // test BoWDescriptor and Colorstructure Descriptor with knn
    int seed = rand();
    descriptor->compareKNN(dataset, classNames[6], 15, 8, 8, 30, seed);

    std::cout << "--------colorStructure-----" << std::endl;
    cv::Ptr<GlobalDescriptor> colorDesc(new ColorStructDesc());
    colorDesc->compareKNN(dataset, classNames[6], 15, 8, 8, 30, seed);
    cv::waitKey(0);

    return 0;
} 
