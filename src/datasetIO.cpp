
#include "datasetIO.hpp"

namespace datasetIO{

	cv::Mat dataItem::getCVMat() 
	{
		return imread(filePath, cv::IMREAD_COLOR);
	}

	//returns environment variable path to dataset
	std::string getCaltechPath()
	{
		char* pPath = getenv("CALTECH_PATH");
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

	//returns string of all folders within directory
	std::vector<std::string> getClassNames(DIR* const caltechTopDir)
	{
		std::vector<std::string> classNames;

		dirent *directoryItem;

		while (caltechTopDir) {

			if ((directoryItem = readdir(caltechTopDir)) != NULL)
			{
				std::string tmp(directoryItem->d_name);
				std::cout << "Item: " << tmp << std::endl;

				if (!(tmp == ".") && !(tmp == ".."))
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

	std::vector <dataItem> getClassSet(const std::string & className, const std::string & caltechPath){
		
		std::vector<dataItem> classSet;

		dirent* directoryItem;

		const std::string currentClassName = className;

		std::string classPath = caltechPath + "/" + currentClassName;
		std::cout << "ClassPath:" << classPath << std::endl;

		DIR* caltechClassDir = opendir(classPath.c_str());

		while (caltechClassDir) {

			if ((directoryItem = readdir(caltechClassDir)) != NULL)
			{
				std::string tmp(directoryItem->d_name);
				std::cout << "Item: " << tmp << std::endl;

				if (!(tmp == ".") && !(tmp == ".."))
				{
					dataItem tmp_dataItem;
					tmp_dataItem.className = currentClassName;
					tmp_dataItem.index = atoi(tmp.substr(6, 4).c_str());
					tmp_dataItem.filePath = classPath + "/" + tmp;
					std::cout << "ClassName: " << tmp_dataItem.className << std::endl;
					std::cout << "Index: " << tmp_dataItem.index << std::endl;
					classSet.push_back(tmp_dataItem);
				}
			}
			else
			{
				closedir(caltechClassDir);
				break;
			}
		}

		return classSet;
	}

	dataSet getDataSet(const std::vector<std::string> & classNames, const std::string & caltechPath)
	{
		std::vector<dataItem> data;
		dirent *directoryItem;
		std::vector<int> classIndices = std::vector<int>();
		classIndices.push_back(0);
		

		for (int currentClassID = 0; currentClassID < classNames.size(); currentClassID++)
		{
			const std::string currentClassName = classNames[currentClassID];

			std::string classPath = caltechPath + "/" + currentClassName;
			std::cout << "ClassPath:" << classPath << std::endl;

			DIR* caltechClassDir = opendir(classPath.c_str());



			while (caltechClassDir) {

				if ((directoryItem = readdir(caltechClassDir)) != NULL)
				{
					std::string tmp(directoryItem->d_name);
					//std::cout << "Item: " << tmp << std::endl;

					if (!(tmp == ".") && !(tmp == ".."))
					{
						dataItem tmp_dataItem;
						tmp_dataItem.className = currentClassName;
						tmp_dataItem.index = atoi(tmp.substr(6, 4).c_str());
						tmp_dataItem.filePath = classPath + "/" + tmp;
						//std::cout << "ClassName: " << tmp_dataItem.className << std::endl;
						//std::cout << "Index: " << tmp_dataItem.index << std::endl;

						data.push_back(tmp_dataItem);
					}
				}
				else
				{
					closedir(caltechClassDir);
					break;
				}
			}

			classIndices.push_back(data.size());
		}

		dataSet dataset;
		dataset.items = data;
		dataset.datasetPath = caltechPath;
		
		for (int currentClassID = 0; currentClassID < classNames.size(); currentClassID++)
		{
			std::vector<dataItem>::iterator begin = data.begin() + classIndices[currentClassID];
			std::vector<dataItem>::iterator end = data.begin() + classIndices[currentClassID + 1];
			std::vector<dataItem> tmpClass = std::vector<dataItem>(begin, end);
			dataset.classDictonary.insert(std::make_pair(classNames[currentClassID], tmpClass));
		}

		return dataset;

	}

}