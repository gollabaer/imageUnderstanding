
#include <globalDescriptors/custHog.hpp>

custHOG::custHOG()
{
    imageSize = 128;
    // has to be <= than the image size !!!!!
    // Detection window size. Align to block size and block stride.
    cv::Size win_size(imageSize, imageSize);

    // Block size in pixels.
    cv::Size block_size(32, 32);

    // Block stride. It must be a multiple of cell size.
    cv::Size block_stride(16, 16);

    // Cell size.
    cv::Size cell_size(16, 16);

    // Number of bins. Only 9 bins per cell are supported for now.
    int nbins(3);

    // Didnt find Usage in Source Code
    // Theoretically Aperture Size of a Sobel Operator
    int derivAperture(1);

    // Gaussian smoothing window parameter.
    double win_sigma(-1);

    int histogramNormType(cv::HOGDescriptor::L2Hys);

    // L2-Hys normalization method shrinkage.
    double threshold_L2hys(0.2);

    // Flag to specify whether the gamma correction preprocessing is required or not.
    bool gamma_correction(true);

    // Maximum number of detection window increases.
    int nlevels(cv::HOGDescriptor::DEFAULT_NLEVELS);

    //hog = cv::HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins,derivAperture,win_sigma,_histogramNormType,threshold_L2hys,gamma_correction,nlevels);
    hog = cv::HOGDescriptor(win_size,block_size,block_stride,
                            cell_size, nbins, derivAperture,win_sigma,
                            histogramNormType,
                            threshold_L2hys, gamma_correction,
                            nlevels);
}

custHOG::~custHOG()
{}

cv::Mat custHOG::compute(datasetIO::dataItem item) const
{
    std::vector<float> descriptors;
    hog.compute(item.getNormedCVMat(imageSize),descriptors);
    return cv::Mat(descriptors,true).t();
}




std::vector<std::string> custHOG::getFeatureDescriptions() const
{
    const int descSize = hog.getDescriptorSize();
    std::vector<std::string> featureDescriptions;
    featureDescriptions.reserve(descSize);

    for(int i = 0; i < descSize; ++i)
    {
        std::stringstream stream;
        stream << "GradOriBin" << i;
        featureDescriptions.push_back(stream.str());
    }
    return featureDescriptions;
}


std::string custHOG::getName() const
{
    return "HOG";
}
