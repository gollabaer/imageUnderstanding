#include <globalDescriptors/ColorStructureDescriptor.hpp>

ColorStructDesc::ColorStructDesc()
{
    descSize = 64;
}


ColorStructDesc::~ColorStructDesc()
{

}

cv::Mat ColorStructDesc::compute(datasetIO::dataItem item) const
{
    cv::Mat mat = item.getNormedCVMat();
    Frame image(mat);
    XM::ColorStructureDescriptor *csd = Feature::getColorStructureD(&image,descSize);

    const int DescriptorLength = csd->GetSize();

    cv::Mat descriptor(1,DescriptorLength,CV_32F);

    for(unsigned int i = 0; i < DescriptorLength; i++)
    {
        const int element = float(csd->GetElement(i));
        descriptor.at<float>(0,i) = element;
    }

    delete csd;
    return descriptor;
}

std::vector<std::string> ColorStructDesc::getFeatureDescriptions() const
{
    std::vector<std::string> featureDescriptions;
    featureDescriptions.reserve(descSize);

    for(int i = 0; i < descSize; ++i)
    {
        std::stringstream stream;
        stream << "quantCol" << i;
        featureDescriptions.push_back(stream.str());
    }
    return featureDescriptions;
}



std::string ColorStructDesc::getName() const
{
    return "ColorStructure";
}
