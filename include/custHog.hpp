#ifndef GLOB_FEAT
#define GLOB_FEAT

#include <opencv2/objdetect/objdetect.hpp>
#include <datasetIO.hpp>

#include <GlobalDescriptor.hpp>

class custHOG : public GlobalDescriptor
{
  cv::HOGDescriptor hog;

public:
  custHOG();
  ~custHOG();

  virtual cv::Mat compute(datasetIO::dataItem item) const;
  virtual cv::Mat compute(std::vector<datasetIO::dataItem> items) const;
};

cv::Mat getRandomHogs(const std::string classname, const unsigned int seed, const int num, datasetIO::dataSet dataset);

void compareHogsOfTwoClasses(const int num, const std::string classname1, const std::string classname2, datasetIO::dataSet dataset, const unsigned int seed);

void compareHogsOfTwoRandomClasses(datasetIO::dataSet dataset, const int numSamples);

void compareHogsOfOneClass(datasetIO::dataSet dataset, std::string className, unsigned int seed);


#endif //GLOB_FEAT
