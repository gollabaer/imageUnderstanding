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


void printParams( cv::Algorithm* algorithm ) {
    std::vector<std::string> parameters;
    algorithm->getParams(parameters);

    for (int i = 0; i < (int) parameters.size(); i++) {
        std::string param = parameters[i];
        int type = algorithm->paramType(param);
        std::string helpText = algorithm->paramHelp(param);
        std::string typeText;

        switch (type) {
        case cv::Param::BOOLEAN:
            typeText = "bool";
            break;
        case cv::Param::INT:
            typeText = "int";
            break;
        case cv::Param::REAL:
            typeText = "real (double)";
            break;
        case cv::Param::STRING:
            typeText = "string";
            break;
        case cv::Param::MAT:
            typeText = "Mat";
            break;
        case cv::Param::ALGORITHM:
            typeText = "Algorithm";
            break;
        case cv::Param::MAT_VECTOR:
            typeText = "Mat vector";
            break;
        }
        std::cout << "Parameter '" << param << "' type=" << typeText << " help=" << helpText << std::endl;
    }
}

namespace descriptor_enum{
enum FeatureDescr
{
    SIFT,
    SURF,
    BRIEF,
    BRISK,
    ORB,
    FREAK
};
}

void testDescriptors(datasetIO::dataSet& dataset)
{
    std::vector<std::string> descriptor_names;
    descriptor_names.push_back("SIFT");
//    descriptor_names.push_back("SURF");
//    descriptor_names.push_back("BRIEF");
//    descriptor_names.push_back("BRISK");
//    descriptor_names.push_back("ORB");
//    descriptor_names.push_back("FREAK");

    int seed = rand();

    // get training images
    std::vector<cv::Mat> trainingImgs;
    for(size_t i = 0; i < dataset.classNames.size(); ++i)
    {
        std::vector<datasetIO::dataItem> trainItems;
        std::vector<datasetIO::dataItem> testItems;
        dataset.getRandomPartionOfClass(dataset.classNames[i], testItems, trainItems, datasetIO::TRAINING_SET_SIZE, seed );
        for(size_t j = 0; j < trainItems.size(); ++j)
        {
            trainingImgs.push_back(trainItems[j].getCVMat());
        }
    }


    // bow Parameters
    cv::initModule_nonfree();
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("ORB");

//    std::cout << "extractor_type: " << extractor->descriptorType() << std::endl;

    cv::TermCriteria termcrit = cv::TermCriteria();
    int attempts = 3;
    int vocabSize = 400;
    int flags = cv::KMEANS_PP_CENTERS;
    bool showKeypoints = false;

    // train Bow with different Extractors
    for(size_t i = 0; i < descriptor_names.size(); ++i)
    {
        std::string descr_name = descriptor_names[i];
        extractor = cv::DescriptorExtractor::create(descr_name);

        // init BoW
        cv::Ptr<BoWDescriptor> bowDescriptor(new BoWDescriptor(detector, extractor, matcher, vocabSize, termcrit, attempts, flags));

        // train BoW
        std::cout << "training BoW " << i+1 << "/" << descriptor_names.size() << std::endl;
        std::stringstream ss;
        ss << "etc/descriptor_test/vocabulary_" << descr_name << ".xml";
        const std::string VOCABULARY_PATH = ss.str();
        bowDescriptor->train(trainingImgs, showKeypoints, VOCABULARY_PATH);
        bowDescriptor->exportTraining_TestDataSetForWEKA(dataset, seed);
    }
}

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

    testDescriptors(dataset);




//     read trained vocabulary from disk
//    std::cout << "read trained vocabulary from disk" << std::endl;
//    bowDescriptor->readVocabularyFromDisk(VOCABULARY_PATH);
//    std::cout << "finished reading" << std::endl;

    // test kmajorityBoWDescriptor and Colorstructure Descriptor with knn
//    int seed = 12;
//    std::cout << "start knn" << std::endl;
//    descriptor->compareKNN(dataset, classNames[9], 15, 8, 8, 30, seed);


//    std::cout << "--------colorStructure-----" << std::endl;
//    cv::Ptr<GlobalDescriptor> colorDesc(new ColorStructDesc());
//    colorDesc->exportDataSetForWEKA(dataset);
//    colorDesc->compareKNN(dataset, classNames[6], 15, 8, 8, 30, seed);




    cv::waitKey(0);

    return 0;
} 
