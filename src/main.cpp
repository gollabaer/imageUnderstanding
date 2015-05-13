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

 
    

	for (int i = 0; i < dataset.classDictonary[classNames[0]].size(); i++)
	{
		cv::namedWindow("rndImg", cv::WINDOW_AUTOSIZE);
		cv::imshow("rndImg", dataset.classDictonary[classNames[0]][i].getCVMat());
		cv::waitKey(100);
	}

    return 0;
} 
