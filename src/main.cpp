// C++-Includes
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* getenv */
#include <dirent.h>
#include <string>
#include <time.h>

// OpenCV-Includes
#include <opencv2/opencv.hpp>

struct dataItem
{
    //std::string path;
    std::string className;
    int index;
    cv::Mat mat;
};

std::string getCaltechPath()
{
    char* pPath = getenv ("CALTECH_PATH");
    if (pPath != NULL)
    {
        std::cout << "CALTECH_PATH is:" << pPath << std::endl;
        return pPath;
    }
    else
    {
        std::cerr << "CALTECH_PATH not set. Exiting ..." << std::endl;
        std::exit(1);
    }

    return std::string();
}

std::vector<std::string> getClassNames(DIR* const caltechTopDir)
{
        std::vector<std::string> classNames;

        dirent *directoryItem;

        while (caltechTopDir) {

            if ((directoryItem = readdir(caltechTopDir)) != NULL)
            {
                std::string tmp(directoryItem->d_name);
                std::cout << "Item: " << tmp << std::endl;

                if(!(tmp == ".") && !(tmp == ".."))
                {
                    std::cout << "Push: " << tmp << std::endl;
                    classNames.push_back(tmp);
                }
            }
            else
            {
                closedir(caltechTopDir);
                break;
            }
        }

        return classNames;
}

std::vector<dataItem> getDataSet(const std::vector<std::string> & classNames, const std::string & caltechPath)
{
    std::vector<dataItem> dataSet;

    dirent *directoryItem;

    for(int currentClassID = 0; currentClassID < classNames.size(); currentClassID++)
    {
        const std::string currentClassName = classNames[currentClassID];

        std::string classPath = caltechPath + "/" + currentClassName;
        std::cout << "ClassPath:" << classPath << std::endl;

        DIR* caltechClassDir = opendir(classPath.c_str());

        while (caltechClassDir) {

            if ((directoryItem = readdir(caltechClassDir)) != NULL)
            {
                std::string tmp(directoryItem->d_name);
                std::cout << "Item: " << tmp << std::endl;

                if(!(tmp == ".") && !(tmp == ".."))
                {
                    dataItem tmp_dataItem;
                    tmp_dataItem.className = currentClassName;
                    tmp_dataItem.index = atoi(tmp.substr(6,4).c_str());
                    tmp_dataItem.mat = cv::imread(classPath + "/" + tmp,  cv::IMREAD_COLOR);
                    std::cout << "ClassName: " << tmp_dataItem.className  << std::endl;
                    std::cout << "Index: " << tmp_dataItem.index << std::endl;
                    std::cout << "Mat.rows: " << tmp_dataItem.mat.rows << std::endl;
                    std::cout << "Mat.cols: " << tmp_dataItem.mat.cols << std::endl;

                    dataSet.push_back(tmp_dataItem);
                }
            }
            else
            {
                closedir(caltechClassDir);
                break;
            }
        }

    }

    return dataSet;

}

int main ()
{

    const std::string caltechPath = getCaltechPath();

    DIR* const caltechTopDir = opendir(caltechPath.c_str());
    if(caltechTopDir == NULL)
    {
        std::cerr << "Error while opening caltech toplevel dir. Exiting ..." << std::endl;
        std::exit(1);
    }

    const std::vector<std::string> classNames = getClassNames(caltechTopDir);

    const std::vector<dataItem> dataSet = getDataSet(classNames, caltechPath);

    srand (time(NULL));
    int rndNum = rand() % dataSet.size();

    cv::namedWindow("rndImg", cv::WINDOW_AUTOSIZE );
    cv::imshow("rndImg",dataSet[rndNum].mat);

    cv::waitKey(0);

    return 0;
} 
