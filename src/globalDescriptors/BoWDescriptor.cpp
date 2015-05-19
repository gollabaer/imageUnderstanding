#include "globalDescriptors/BoWDescriptor.hpp"

BoWDescriptor::BoWDescriptor(const cv::Ptr<cv::DescriptorExtractor>& dextractor, const cv::Ptr<cv::DescriptorMatcher>& dmatcher,
                             int clusterCount, const cv::TermCriteria& termcrit, int attempts, int flags)
    :
      m_bowTrainer(cv::BOWKMeansTrainer(clusterCount, termcrit, attempts, flags)),
      m_bowExtractor(dextractor, dmatcher),
      m_featureExtractor(dextractor),
      m_featureMatcher(dmatcher),
      m_trained(false),
      m_featureDetector(cv::FeatureDetector::create("SURF"))
{

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
    m_featureDetector->detect(image,keypoints);
    m_bowExtractor.compute(image,keypoints,imageDescriptor);

    return imageDescriptor;
}

void BoWDescriptor::train(const std::vector<cv::Mat>& trainImages)
{
    std::vector<std::vector<cv::KeyPoint> > keypoints_vec;
    m_featureDetector->detect(trainImages,keypoints_vec);
    std::vector<cv::Mat> descriptors_vec;
    m_featureExtractor->compute(trainImages, keypoints_vec, descriptors_vec);

    for(size_t i = 0; i < descriptors_vec.size(); ++i)
    {
        if(descriptors_vec[i].type() == CV_32F)
        {
            m_bowTrainer.add(descriptors_vec[i]);
        }
        else
        {
            std::cerr << "bad descriptor type" << std::endl;
        }
    }
    cv::Mat vocabulary = m_bowTrainer.cluster();
    m_bowExtractor.setVocabulary(vocabulary);

    m_trained = true;
}
