#ifndef BOWDESCRIPTOR_HPP
#define BOWDESCRIPTOR_HPP

#include <GlobalDescriptor.hpp>
#include <opencv2/nonfree/nonfree.hpp>

class BoWDescriptor : public GlobalDescriptor
{
private:
    BoWDescriptor();
    cv::BOWKMeansTrainer m_bowTrainer;
    // needs to be mutable since its compute() method is not const
    mutable cv::BOWImgDescriptorExtractor m_bowExtractor;

    cv::Ptr<cv::FeatureDetector> m_featureDetector;
    cv::Ptr<cv::DescriptorExtractor> m_featureExtractor;
    cv::Ptr<cv::DescriptorMatcher> m_featureMatcher;

    bool m_trained;

public:
    BoWDescriptor(const cv::Ptr<cv::DescriptorExtractor> &dextractor, const cv::Ptr<cv::DescriptorMatcher> &dmatcher,
                  int clusterCount, const cv::TermCriteria& termcrit = cv::TermCriteria(), int attempts = 3, int flags = cv::KMEANS_PP_CENTERS );
    virtual ~BoWDescriptor();

    virtual cv::Mat compute(datasetIO::dataItem item) const;
    void train(const std::vector<cv::Mat>& trainImages);
};

#endif // BOWDESCRIPTOR_HPP
