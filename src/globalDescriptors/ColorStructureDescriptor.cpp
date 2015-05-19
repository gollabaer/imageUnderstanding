#include <globalDescriptors/ColorStructureDescriptor.hpp>

ColorStructDesc::ColorStructDesc()
{

}


ColorStructDesc::~ColorStructDesc()
{

}

cv::Mat ColorStructDesc::compute(datasetIO::dataItem item) const
{
    cv::Mat mat = item.getNormedCVMat();
    Frame image(mat);
    XM::ColorStructureDescriptor *csd = Feature::getColorStructureD(&image);

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

