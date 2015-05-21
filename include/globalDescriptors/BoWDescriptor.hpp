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
    cv::Mat m_vocabulary;

    // writes trained vocabulary using cv::Filestorage
    void writeVocabularyToDisk(std::string filepath) const;
    void visualizeKeypoints(const std::vector<cv::Mat>& images, const std::vector<std::vector<cv::KeyPoint> > &keypoints_vec, size_t wait = 0) const;

    bool m_trained;

public:
    BoWDescriptor(const cv::Ptr<cv::DescriptorExtractor> &dextractor, const cv::Ptr<cv::DescriptorMatcher> &dmatcher,
                  int clusterCount, const cv::TermCriteria& termcrit = cv::TermCriteria(), int attempts = 3, int flags = cv::KMEANS_PP_CENTERS );
    virtual ~BoWDescriptor();

    // compute the Histogram of trained visual Words for a query image
    virtual cv::Mat compute(datasetIO::dataItem item) const;


    std::string getName() const;

    // train the vocabulary of visual words
    void train(const std::vector<cv::Mat>& trainImages, bool debugVis = false);
    bool readVocabularyFromDisk(std::string filepath);
    bool isTrained() const;
};

#endif // BOWDESCRIPTOR_HPP
