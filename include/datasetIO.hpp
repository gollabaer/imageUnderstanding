
#ifndef DATASET_IO
#define DATASET_IO

#include <stdio.h>
#include <string>
#include <vector>
#include <dirent.h>

//OpenCV includes
#include <opencv2/opencv.hpp>


namespace datasetIO{

	struct dataItem
	{
		std::string filePath;
		std::string imageName;
		std::string className;
		int index;

        cv::Mat getCVMat() const;
        cv::Mat getNormedCVMat(int size = 150) const;
	};

	struct dataSet
	{
		std::string datasetPath;
		std::vector<dataItem> items;
		std::map<std::string, std::vector<dataItem> > classDictonary;
        std::vector<std::string> classNames;

        std::vector<dataItem> getItemsFromClass(std::string className);

        std::vector<cv::Mat> getRandomImagesFromClass(int num, std::string className, unsigned int seed) const;
        std::vector<cv::Mat> getRandomNormedImagesFromClass(int num, std::string className, unsigned int seed, int size = 150) const;
        std::vector<dataItem> getRandomItemsFromClass(int num, std::string className, unsigned int seed) const;
		void slideShow(std::string className = std::string("none"), int waitKey = 200, bool normed = true);
    };

	std::string getCaltechPath();


	std::vector<std::string> getClassNames(DIR* const caltechTopDir);

	std::vector <dataItem> getClassSet(const std::string & className, const std::string & caltechPath);

	dataSet getDataSet(const std::vector<std::string> & classNames, const std::string & caltechPath);


}

#endif DATASET_IO
