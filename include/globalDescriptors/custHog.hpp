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

#endif //GLOB_FEAT
