// C++-Includes
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* getenv */
#include <dirent.h>
#include <string>
#include <time.h>

// OpenCV-Includes
#include <opencv2/opencv.hpp>

#include "datasetIO.hpp"




int main ()
{

    const std::string caltechPath = datasetIO::getCaltechPath();

    DIR* const caltechTopDir = opendir(caltechPath.c_str());
    if(caltechTopDir == NULL)
    {
        std::cerr << "Error while opening caltech toplevel dir. Exiting ..." << std::endl;
        std::exit(1);
    }

    const std::vector<std::string> classNames = datasetIO::getClassNames(caltechTopDir);

     datasetIO::dataSet dataset = datasetIO::getDataSet(classNames, caltechPath);

    std::vector<cv::Mat> randomImgs = dataset.getRandomImagesFromClass(10,"binocular",5);

    for(int i = 0; i < randomImgs.size(); ++i)
    {
        cv::namedWindow("rndImg" + i, cv::WINDOW_AUTOSIZE);
        cv::imshow("rndImg" + i, randomImgs[i]);
        cv::waitKey(10);
    }

    cv::waitKey(0);

    return 0;
} 
