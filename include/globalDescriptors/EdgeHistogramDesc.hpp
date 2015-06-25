#ifndef _EHD_
#define _EHD_

#include <Feature.h>
#include <datasetIO.hpp>
#include <GlobalDescriptor.hpp>

class EdgeHistogramDesc : public GlobalDescriptor
{

public:
  EdgeHistogramDesc();
  ~EdgeHistogramDesc();

  virtual cv::Mat compute(datasetIO::dataItem item) const;

  std::vector<std::string> getFeatureDescriptions() const;

  std::string getName() const;

private:
  int descSize;
};

#endif //_EHD_
