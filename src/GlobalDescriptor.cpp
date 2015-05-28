#include <GlobalDescriptor.hpp>

#include <iostream>
#include <fstream>

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


    if(dataset.classDictonary.find(className) != dataset.classDictonary.end())
    {
        const std::vector<datasetIO::dataItem> classItems = dataset.classDictonary.at(className);
        std::vector<datasetIO::dataItem> classItemCopy = classItems;
        std::vector<datasetIO::dataItem> classItemCopy2;

        const int size = classItems.size()/2;

        dataset.getRandomPartionOfClass(className,classItemCopy,classItemCopy2, size,seed);

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

void GlobalDescriptor::compareKNN(datasetIO::dataSet dataset, const std::string className, const int testSampleSize, const int k,
                                  const int numAdditionalClasses, const int samplesPerClass,unsigned int seed)
{
    srand(seed);

    cv::FlannBasedMatcher flann;

    std::vector<std::string> currentClassDict;

    currentClassDict.push_back(className);

    std::vector<std::string> exc; exc.push_back(className);

    std::vector<std::string> randClasses = dataset.getRandomClasses(numAdditionalClasses,rand(),exc);

    for(int i = 0; i < randClasses.size(); ++i)
    {
        currentClassDict.push_back(randClasses[i]);
    }

    std::vector<cv::Mat> descriptors;

    std::vector<datasetIO::dataItem> testSamples = dataset.getRandomItemsFromClass(testSampleSize,className,rand());
    std::vector<datasetIO::dataItem> samplesFromSameClass = dataset.getRandomItemsFromClass(samplesPerClass,className,rand());

    descriptors.push_back(this->compute(samplesFromSameClass));
    for(int i = 1; i < currentClassDict.size(); ++i)
    {
        const std::string currentClassName = currentClassDict[i];
        const std::vector<datasetIO::dataItem> itemsFromClass = dataset.getRandomItemsFromClass(samplesPerClass,currentClassName,rand());

        const cv::Mat DescriptorsOfClass = this->compute(itemsFromClass);

        descriptors.push_back(DescriptorsOfClass);
    }
    flann.add(descriptors);
    flann.train();

    const cv::Mat queryDescs = this->compute(testSamples);
    std::vector<std::vector<cv::DMatch> > matches;
    flann.knnMatch(queryDescs,matches,k);

    std::cout << "KNN Test BEGIN" << std::endl;
    std::cout << "used classes" << std::endl;
    for(size_t i = 0; i < randClasses.size(); ++i)
    {
       std::cout << randClasses[i] << std::endl;
    }
    std::cout << std::endl;

    for(int i = 0; i < matches.size(); ++i)
    {
        const std::vector<cv::DMatch> currentKNNMatch = matches[i];
        std::cout << "Image " << testSamples[currentKNNMatch[0].queryIdx].index << " of " << className << ". Votes:" << std::endl;
        for(int j = 0; j < currentKNNMatch.size(); ++j)
        {
            const cv::DMatch currentMatch = currentKNNMatch[j];
            std::cout << currentClassDict[currentMatch.imgIdx] << " --- distance: " << currentMatch.distance << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "KNN Test END" << std::endl;
}

void GlobalDescriptor::exportSelectedItemsForWEKA(std::vector<datasetIO::dataItem> items_to_export, std::vector<std::string> classNames_to_export, std::string filename)
{
    std::string atRelation = "@RELATION caltech101";
    std::string atAttribute = "@ATTRIBUTE";
    std::string atData = "@DATA";
    std::string datatypeNumeric = "NUMERIC";
    std::string tab = "\t";
    std::vector<std::string> featureDescriptions = this->getFeatureDescriptions();

    std::ofstream outfile;
    outfile.open(filename.c_str());
    outfile << atRelation << std::endl;
    outfile << std::endl;

    for(int i = 0; i<featureDescriptions.size();++i)
    {
        outfile << atAttribute << tab << featureDescriptions[i] << tab << datatypeNumeric << std::endl;
    }

    outfile << atAttribute << tab << "class" << tab << "{";

    for(int i = 0; i < classNames_to_export.size(); ++i)
    {
        outfile << classNames_to_export[i];
        if(i < classNames_to_export.size() - 1)
            outfile << ",";
    }

    outfile << "}" << std::endl;

    outfile << std::endl;
    outfile << atData << std::endl;

    for(int i = 0; i < items_to_export.size(); ++i)
    {
        std::cout << "item: " << i << " of " << items_to_export.size() << std::endl;
        const datasetIO::dataItem item = items_to_export[i];
        const cv::Mat desc = this->compute(item);

        for(int j = 0; j < desc.cols; ++j)
        {
            switch(desc.type())
            {
            case CV_8U: outfile << desc.at<uchar>(0,j);break;
            case CV_8S: outfile << desc.at<char>(0,j);break;
            case CV_16U:outfile << desc.at<ushort>(0,j);break;
            case CV_16S:outfile << desc.at<short>(0,j);break;
            case CV_32S:outfile << desc.at<int>(0,j);break;
            case CV_32F:outfile << desc.at<float>(0,j);break;
            case CV_64F:outfile << desc.at<double>(0,j);break;
            default: std::cout << "[exportToWeka] what?" << std::endl;break;
            }
            outfile << ",";
        }
        outfile << item.className << std::endl;
    }
}

void GlobalDescriptor::exportDataSetForWEKA(const datasetIO::dataSet dataset)
{
    std::vector<datasetIO::dataItem> items_to_export = dataset.items;
    std::vector<std::string> classNames_to_export = dataset.classNames;
    std::string filename = (std::string("etc/") + this->getName() + "_WEKA.arff");

    exportSelectedItemsForWEKA(items_to_export, classNames_to_export, filename);
}

void GlobalDescriptor::exportTraining_TestDataSetForWEKA(const datasetIO::dataSet dataset, const unsigned int seed)
{
    const int itemsPerClassFortraining = 15;
    std::vector<datasetIO::dataItem> test_items;
    std::vector<datasetIO::dataItem> train_items;
    std::vector<std::string> classNames_to_export = dataset.classNames;

    std::stringstream stream1;
    stream1 << "etc/" << this->getName() << "_train_seed_" << seed << "_WEKA.arff";
    std::string filename_training = stream1.str();

    std::stringstream stream2;
    stream2 << "etc/" << this->getName() << "_test_seed_" << seed << "_WEKA.arff";
    std::string filename_test = stream2.str();

    for(int i = 0; i < classNames_to_export.size(); ++i)
    {
        std::vector<datasetIO::dataItem> part1;
        std::vector<datasetIO::dataItem> part2;
        dataset.getRandomPartionOfClass(classNames_to_export[i],part1,part2,itemsPerClassFortraining,seed);

        test_items.reserve(test_items.size() + part1.size());
        test_items.insert(test_items.end(),part1.begin(),part1.end());

        train_items.reserve(train_items.size() + part2.size());
        train_items.insert(train_items.end(),part2.begin(),part2.end());
    }
    exportSelectedItemsForWEKA(train_items,classNames_to_export,filename_training);
    exportSelectedItemsForWEKA(test_items,classNames_to_export,filename_test);
}

