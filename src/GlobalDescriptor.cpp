#include <GlobalDescriptor.hpp>

cv::Mat GlobalDescriptor::compute(std::vector<datasetIO::dataItem> items) const
{
    cv::Mat comb_res;
    for(int i = 0; i < items.size(); ++i)
    {
        comb_res.push_back(compute(items[i]));
    }

    return comb_res;
}

cv::Mat GlobalDescriptor::getRandomImageInClassDescriptors(const std::string classname,
                                         const unsigned int seed, const int num, datasetIO::dataSet dataset)
{
    std::vector<datasetIO::dataItem> randomItemsFromClass = dataset.getRandomItemsFromClass(num, classname, seed);
    cv::Mat comb_res;

    for(int i = 0; i < randomItemsFromClass.size(); ++i)
    {
        comb_res.push_back(this->compute(randomItemsFromClass[i]));
    }

    return comb_res;
}

void GlobalDescriptor::compareDescriptorsOfTwoClasses(const int num, const std::string classname1, const std::string classname2,
                                                      datasetIO::dataSet dataset, const unsigned int seed)
{
    srand(seed);
    cv::Mat descriptorsOfClass1 = getRandomImageInClassDescriptors(classname1, rand(), num, dataset);
    cv::Mat descriptorsOfClass2 = getRandomImageInClassDescriptors(classname2, rand(), num, dataset);

    cv::FlannBasedMatcher flann;

    std::vector<cv::DMatch> matches;
    flann.match(descriptorsOfClass1,descriptorsOfClass2,matches);

    float avgDist(0.0);
    for(int i = 0; i < matches.size(); ++i)
    {
        avgDist += matches[i].distance;
    }
    avgDist /= float(matches.size());
    std::cout << "Average Distance between " << classname1 << " and " << classname2 << " is " << avgDist << ". Samples " << matches.size() << "." << std::endl;
    compareDescriptorsOfOneClass(dataset, classname1,rand());
    compareDescriptorsOfOneClass(dataset, classname2,rand());
}

void GlobalDescriptor::compareDescriptorsOfTwoRandomClasses(datasetIO::dataSet dataset, const int numSamples)
{
    srand(time(NULL));
    std::string class1 = dataset.classNames[rand() % dataset.classNames.size()];
    std::string class2 = dataset.classNames[rand() % dataset.classNames.size()];

    compareDescriptorsOfTwoClasses(numSamples, class1, class2, dataset, rand());
}


void GlobalDescriptor::compareDescriptorsOfOneClass(datasetIO::dataSet dataset, std::string className, unsigned int seed)
{
    std::vector<datasetIO::dataItem> randomItems;
    if(dataset.classDictonary.find(className) != dataset.classDictonary.end())
    {
        const std::vector<datasetIO::dataItem> classItems = dataset.classDictonary.at(className);
        std::vector<datasetIO::dataItem> classItemCopy = classItems;

        std::srand(seed);

        std::vector<datasetIO::dataItem> classItemCopy2;
        classItemCopy2.reserve(classItems.size());

        std::vector<int> randNumMemory;
        for(int i = 0; i < classItems.size()/2;)
        {
            int randNum = std::rand() % classItemCopy.size();
            if(std::find(randNumMemory.begin(),randNumMemory.end(),randNum) == randNumMemory.end())
            {
                randNumMemory.push_back(randNum);
                ++i;

                classItemCopy2.push_back(classItemCopy[randNum]);
                classItemCopy.erase(classItemCopy.begin() + randNum);
            }
        }

        cv::Mat descriptors1 = this->compute(classItemCopy);

        cv::Mat descriptors2 = this->compute(classItemCopy2);

        cv::FlannBasedMatcher flann;

        std::vector<cv::DMatch> matches;
        flann.match(descriptors1,descriptors2,matches);

        float avgDist(0.0);
        for(int i = 0; i < matches.size(); ++i)
        {
            avgDist += matches[i].distance;
        }
        avgDist /= float(matches.size());
        std::cout << "Average Distance in class " << className << " is " << avgDist << ". Samples " << matches.size() << "." << std::endl;
        return;
    }

    std::cout << "Wrong class." << std::endl;
}

