
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
	};

	struct dataSet
	{
		std::string datasetPath;
		std::vector<dataItem> items;
		std::map<std::string, std::vector<dataItem> > classDictonary;

        std::vector<cv::Mat> getRandomImagesFromClass(int num, std::string className, unsigned int seed) const;
	};

	std::string getCaltechPath();


	std::vector<std::string> getClassNames(DIR* const caltechTopDir);

	std::vector <dataItem> getClassSet(const std::string & className, const std::string & caltechPath);

	dataSet getDataSet(const std::vector<std::string> & classNames, const std::string & caltechPath);


}
