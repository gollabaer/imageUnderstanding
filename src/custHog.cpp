
#include <custHog.hpp>

custHOG::custHOG()
{

    // has to be <= than the image size !!!!!
    // Detection window size. Align to block size and block stride.
    cv::Size win_size(128, 128);

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

cv::Mat custHOG::computeHOG(datasetIO::dataItem item)
{
    std::vector<float> descriptors;
    hog.compute(item.getNormedCVMat(128),descriptors);
    return cv::Mat(descriptors,true).t();
}

cv::Mat getRandomHogs(const std::string classname, const unsigned int seed, const int num, datasetIO::dataSet dataset)
{
    custHOG hog;
    std::vector<datasetIO::dataItem> randomItemsFromClass = dataset.getRandomItemsFromClass(num,classname,seed);
    cv::Mat comb_res;

    for(int i = 0; i < randomItemsFromClass.size(); ++i)
    {
        comb_res.push_back(hog.computeHOG(randomItemsFromClass[i]));
    }

    return comb_res;
}

void compareHogsOfTwoClasses(const int num, const std::string classname1, const std::string classname2, datasetIO::dataSet dataset, const unsigned int seed)
{
    srand(seed);
    cv::Mat hogsOfClass1 = getRandomHogs(classname1, rand(), num, dataset);
    cv::Mat hogsOfClass2 = getRandomHogs(classname2,rand(),num,dataset);

    cv::FlannBasedMatcher flann;

    std::vector<cv::DMatch> matches;
    flann.match(hogsOfClass1,hogsOfClass2,matches);

    float avgDist(0.0);
    for(int i = 0; i < matches.size(); ++i)
    {
        avgDist += matches[i].distance;
    }
    avgDist /= float(matches.size());
    std::cout << "Average Distance between " << classname1 << " and " << classname2 << " is " << avgDist << ". Samples " << matches.size() << "." << std::endl;
}


void compareHogsOfTwoRandomClasses(datasetIO::dataSet dataset, const int numSamples)
{
    srand(time(NULL));
    std::string class1 = dataset.classNames[rand() % dataset.classNames.size()];
    std::string class2 = dataset.classNames[rand() % dataset.classNames.size()];

    compareHogsOfTwoClasses(numSamples,class1,class2,dataset,rand());
}


void compareHogsOfOneClass(datasetIO::dataSet, std::string className)
{

}
