#ifndef GLOBALDESCRIPTOR_H
#define GLOBALDESCRIPTOR_H

#include <datasetIO.hpp>
#include <opencv2/opencv.hpp>

class GlobalDescriptor{
public:
    // compute a descriptor for one item
    virtual cv::Mat compute(datasetIO::dataItem item) const = 0;
    // compute a descriptor for each item in items
    virtual cv::Mat compute(std::vector<datasetIO::dataItem> items) const = 0;
    virtual ~GlobalDescriptor(){}

    // get a descriptor of num random images in the class specified by classname from the dataset
    cv::Mat getRandomImageInClassDescriptors(const std::string classname, const unsigned int seed,
                                             const int num, datasetIO::dataSet dataset);
    void compareDescriptorsOfTwoClasses(const int num, const std::string classname1, const std::string classname2,
                                 datasetIO::dataSet dataset, const unsigned int seed);
    void compareDescriptorsOfTwoRandomClasses(datasetIO::dataSet dataset, const int numSamples);

    void compareDescriptorsOfOneClass(datasetIO::dataSet dataset, std::string className, unsigned int seed);
};


#endif // GLOBALDESCRIPTOR_H
