#ifndef DOM_COLOR
#define DOM_COLOR

#include <Feature.h>
#include <datasetIO.hpp>

#include <GlobalDescriptor.hpp>

class ColorStructDesc : public GlobalDescriptor
{

public:
  ColorStructDesc();
  ~ColorStructDesc();

  virtual cv::Mat compute(datasetIO::dataItem item) const;

  std::vector<std::string> getFeatureDescriptions() const;

  std::string getName() const;

private:
  int descSize;

};

#endif //DOM_COLOR
