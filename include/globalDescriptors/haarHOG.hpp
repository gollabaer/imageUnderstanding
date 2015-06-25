#ifndef _HAAR_HOG_
#define _HAAR_HOG_

#include <globalDescriptors/custHog.hpp>

class HaarHOG : public custHOG
{


    // GlobalDescriptor interface
public:
    cv::Mat compute(datasetIO::dataItem item) const;
    std::string getName() const;
};

#endif //_HAAR_HOG_
