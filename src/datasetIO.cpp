
#include "datasetIO.hpp"

namespace datasetIO{

    cv::Mat dataItem::getCVMat() const
	{
		return imread(filePath, cv::IMREAD_COLOR);
	}
    cv::Mat dataItem::getNormedCVMat(int size) const
    {
        cv::Mat img = getCVMat();
        const int height = img.rows;
        const int width = img.cols;

        const int min_size = std::min(height,width);

        cv::Rect roi;
        roi.x = width/2 - min_size/2;
        roi.y = height/2 -min_size/2;
        roi.width = min_size;
        roi.height = min_size;

        cv::Mat roiImg;

        roiImg = img(roi).clone();

        resize(roiImg, roiImg, cv::Size(size,size));

        return roiImg;
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

				if (!(tmp == ".") && !(tmp == ".."))
				{
                    std::cout << "Add Class: " << tmp << std::endl;
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

			DIR* caltechClassDir = opendir(classPath.c_str());



			while (caltechClassDir) {

				if ((directoryItem = readdir(caltechClassDir)) != NULL)
				{
					std::string tmp(directoryItem->d_name);

					if (!(tmp == ".") && !(tmp == ".."))
					{
						dataItem tmp_dataItem;
						tmp_dataItem.className = currentClassName;
						tmp_dataItem.index = atoi(tmp.substr(6, 4).c_str());
						tmp_dataItem.filePath = classPath + "/" + tmp;

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

    std::vector<dataItem> dataSet::getItemsFromClass(std::string className)
    {
        if(classDictonary.find(className) != classDictonary.end())
        {
            return classDictonary.at(className);
        }
        else
        {
            std::cout << "[getItemsFromClass] No Class named: " << className;
            return std::vector<dataItem>();
        }
    }

    std::vector<cv::Mat> dataSet::getRandomImagesFromClass(int num, std::string className, unsigned int seed) const
    {
        std::vector<cv::Mat> randomImages;
        std::vector<dataItem> randomItemsFromClass = getRandomItemsFromClass(num,className,seed);
        for(int i=0; i < randomItemsFromClass.size(); ++i)
        {
            randomImages.push_back(randomItemsFromClass[i].getCVMat());
        }
        return randomImages;
    }

    std::vector<cv::Mat> dataSet::getRandomNormedImagesFromClass(int num, std::string className, unsigned int seed, int size) const
    {
        std::vector<cv::Mat> randomImages;
        std::vector<dataItem> randomItemsFromClass = getRandomItemsFromClass(num,className,seed);
        for(int i=0; i < randomItemsFromClass.size(); ++i)
        {
            randomImages.push_back(randomItemsFromClass[i].getNormedCVMat(size));
        }
        return randomImages;
    }

    std::vector<dataItem> dataSet::getRandomItemsFromClass(int num, std::string className, unsigned int seed) const
    {
        std::vector<dataItem> randomItems;
        if(classDictonary.find(className) != classDictonary.end())
        {
            const std::vector<dataItem> classItems = classDictonary.at(className);

            if(num > classItems.size())
            {
                for(int i = 0; i < classItems.size(); ++i)
                {
                    // "random"
                    randomItems.push_back(classItems[0]);
                }
                std::cout << "[getRandomItemsFromClass] you wanted more items than the class has";
                return randomItems;
            }

            std::srand(seed);

            std::vector<int> randNumMemory;
            for(int i = 0; i < num;)
            {
                int randNum = std::rand() % classItems.size();
                if(std::find(randNumMemory.begin(),randNumMemory.end(),randNum) == randNumMemory.end())
                {
                    randNumMemory.push_back(randNum);
                    ++i;

                    randomItems.push_back(classItems[randNum]);
                }
            }

            return randomItems;
        }

        std::cout << "[getRandomItemsFromClass] No Class named: " << className;
        return randomItems;
    }

	void dataSet::slideShow(std::string className, int waitKey , bool normed ){
		if (normed){
			if (className == std::string("none")){

				for (int i = 0; i < items.size(); i++){

					cv::namedWindow(items[i].className, cv::WINDOW_NORMAL);
					cv::imshow(items[i].className, items[i].getNormedCVMat());
					cv::waitKey(waitKey);
				}
			}
			else{

				for (int i = 0; i < classDictonary[className].size(); i++){
					cv::namedWindow(classDictonary[className][i].className, cv::WINDOW_NORMAL);
					cv::imshow(classDictonary[className][i].className, classDictonary[className][i].getNormedCVMat());
					cv::waitKey(waitKey);
				}

			}
		}
		else{
			if (className == std::string("none")){

				for (int i = 0; i < items.size(); i++){

					cv::namedWindow(items[i].className, cv::WINDOW_AUTOSIZE);
					cv::imshow(items[i].className, items[i].getCVMat());
					cv::waitKey(waitKey);
				}
			}
			else{

				for (int i = 0; i < classDictonary[className].size(); i++){
					cv::namedWindow(classDictonary[className][i].className, cv::WINDOW_AUTOSIZE);
					cv::imshow(classDictonary[className][i].className, classDictonary[className][i].getCVMat());
					cv::waitKey(waitKey);
				}

			}


		}
	}
	
}
