#ifndef GLOB_FEAT
#define GLOB_FEAT

#include <opencv2/objdetect/objdetect.hpp>
#include <datasetIO.hpp>

class custHOG{
  cv::HOGDescriptor hog;

public:
  custHOG();

  cv::Mat computeHOG(datasetIO::dataItem item);
  cv::Mat computeHOGs(std::vector<datasetIO::dataItem> items);
};

cv::Mat getRandomHogs(const std::string classname, const unsigned int seed, const int num, datasetIO::dataSet dataset);

void compareHogsOfTwoClasses(const int num, const std::string classname1, const std::string classname2, datasetIO::dataSet dataset, const unsigned int seed);

void compareHogsOfTwoRandomClasses(datasetIO::dataSet dataset, const int numSamples);

void compareHogsOfOneClass(datasetIO::dataSet dataset, std::string className, unsigned int seed);


#endif //GLOB_FEAT
