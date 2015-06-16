#include "globalDescriptors/BoWDescriptor.hpp"

BoWDescriptor::BoWDescriptor(const cv::Ptr<cv::FeatureDetector> &detector, const cv::Ptr<cv::DescriptorExtractor>& dextractor, const cv::Ptr<cv::DescriptorMatcher>& dmatcher,
                             int clusterCount, const cv::TermCriteria& termcrit, int attempts, int flags)
    :
      m_vocabSize(clusterCount),
      m_bowTrainer(cv::BOWKMeansTrainer(m_vocabSize, termcrit, attempts, flags)),
      m_bowExtractor(dextractor, dmatcher),
      m_featureExtractor(dextractor),
      m_featureMatcher(dmatcher),
      m_trained(false),
      m_featureDetector(detector)
{

}

void BoWDescriptor::writeVocabularyToDisk(std::string filepath) const
{
    if(!m_trained)
    {
        std::cerr << "BoW error: no vocabulary to write" << std::endl;
        return;
    }
    cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
    fs << "vocabulary" << m_vocabulary;
    fs.release();
}

bool BoWDescriptor::readVocabularyFromDisk(std::string filepath)
{
    cv::FileStorage fs;
    if(!fs.open(filepath, cv::FileStorage::READ))
    {
        std::cerr << "BoW error: can not find " << filepath << std::endl;
        return false;
    }
    fs["vocabulary"] >> m_vocabulary;
    fs.release();
    m_bowExtractor.setVocabulary(m_vocabulary);
    m_trained = true;
}

bool BoWDescriptor::isTrained() const
{
    return m_trained;
}

BoWDescriptor::~BoWDescriptor()
{

}

cv::Mat BoWDescriptor::compute(datasetIO::dataItem item) const
{
    if(!m_trained)
    {
        return cv::Mat();
    }
    cv::Mat imageDescriptor;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat image = item.getCVMat();
    m_featureDetector->detect(image, keypoints);
    m_bowExtractor.compute(image, keypoints, imageDescriptor);

    return imageDescriptor;
}

std::vector<std::string> BoWDescriptor::getFeatureDescriptions() const
{
    const int descSize = m_vocabSize;
    std::vector<std::string> featureDescriptions;
    featureDescriptions.reserve(descSize);

    for(int i = 0; i < descSize; ++i)
    {
        std::stringstream stream;
        stream << "word" << i;
        featureDescriptions.push_back(stream.str());
    }
    return featureDescriptions;
}

void BoWDescriptor::visualizeKeypoints(const std::vector<cv::Mat> &images, const std::vector<std::vector<cv::KeyPoint> > &keypoints_vec, size_t wait) const
{
    for(size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat debugImg = images[i].clone();
        cv::drawKeypoints(debugImg, keypoints_vec[i], debugImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("keypoints", cv::WINDOW_NORMAL);
        cv::imshow("keypoints", debugImg);
        cv::waitKey(wait);
    }
}

void BoWDescriptor::train(const std::vector<cv::Mat>& trainImages, bool debugVis)
{
    std::vector<std::vector<cv::KeyPoint> > keypoints_vec;
    m_featureDetector->detect(trainImages,keypoints_vec);

    if(debugVis)
    {
        visualizeKeypoints(trainImages, keypoints_vec);
    }

    std::vector<cv::Mat> descriptors_vec;
    m_featureExtractor->compute(trainImages, keypoints_vec, descriptors_vec);

    for(size_t i = 0; i < descriptors_vec.size(); ++i)
    {
        cv::Mat tmp;
        descriptors_vec[i].convertTo(tmp, CV_32F);
        if(tmp.type() == CV_32F)
        {
            m_bowTrainer.add(tmp);
        }
        else
        {
            std::cerr << "bad descriptor type" << std::endl;
        }
    }
    m_vocabulary = m_bowTrainer.cluster();
    m_bowExtractor.setVocabulary(m_vocabulary);

    m_trained = true;
    writeVocabularyToDisk("etc/vocabulary.xml");
}


std::string BoWDescriptor::getName() const
{
    return "BOW";
}
