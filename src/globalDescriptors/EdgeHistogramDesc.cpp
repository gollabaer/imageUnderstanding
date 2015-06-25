#include "globalDescriptors/EdgeHistogramDesc.hpp"

EdgeHistogramDesc::EdgeHistogramDesc()
{
    descSize = 80;
}

EdgeHistogramDesc::~EdgeHistogramDesc()
{

}

cv::Mat EdgeHistogramDesc::compute(datasetIO::dataItem item) const
{
    cv::Mat mat = item.getNormedCVMat();
    Frame image(mat);
    XM::EdgeHistogramDescriptor *ehd = Feature::getEdgeHistogramD(&image);

    const int DescriptorLength = descSize;
    double * desc = ehd->GetEdgeHistogramD();

    cv::Mat descriptor(1,DescriptorLength,CV_64F);

    for(unsigned int i = 0; i < DescriptorLength; i++)
    {
        const double & element = desc[i];
        descriptor.at<double>(0,i) = element;
    }

    delete ehd;
    return descriptor;
}

std::vector<std::string> EdgeHistogramDesc::getFeatureDescriptions() const
{
    std::vector<std::string> featureDescriptions;
    featureDescriptions.reserve(descSize);

    for(int i = 0; i < descSize; ++i)
    {
        std::stringstream stream;
        stream << "quantEdgeDir" << i;
        featureDescriptions.push_back(stream.str());
    }
    return featureDescriptions;
}

std::string EdgeHistogramDesc::getName() const
{
    return "EdgeHistogramm";
}
