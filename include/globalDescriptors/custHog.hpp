#ifndef GLOB_FEAT
#define GLOB_FEAT

#include <opencv2/objdetect/objdetect.hpp>
#include <datasetIO.hpp>

#include <GlobalDescriptor.hpp>

class custHOG : public GlobalDescriptor
{
  cv::HOGDescriptor hog;
  int imageSize;

public:
  custHOG();
  ~custHOG();

  virtual cv::Mat compute(datasetIO::dataItem item) const;

  std::vector<std::string> getFeatureDescriptions() const;

  std::string getName() const;
};

#endif //GLOB_FEAT
