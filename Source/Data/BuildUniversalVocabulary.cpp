#include <Feature/ColorHistogram.hpp>
#include <Feature/HistogramOfOrientedGradients.hpp>
#include <BagOfFeatures/Codewords.hpp>
#include <Quantization/VocabularyTreeQuantization.hpp>
#include <Util/Clustering.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <map>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace ColorTextureShape;
using namespace LocalDescriptorAndBagOfFeature;

cv::Size regionSize(54, 54); // Default HoG is 3x3 cell blocks of 6x6 pixel cells, this gives 3x3 block regions

std::vector<cv::Mat> load_images(std::string imageFileList)
{
    std::ifstream iss(imageFileList);

    std::vector<cv::Mat> images;
    while(iss)
    {
        std::string imFile;
        std::getline(iss,imFile);

        cv::Mat img = cv::imread(imFile);
        if(img.data != NULL)
            images.push_back(img);
    }

    return images;
}

int main(int argc, char **argv)
{
    // Feature set
    std::vector<HistogramFeature *> features;
    features.push_back(new HistogramOfOrientedGradients());
    features.push_back(new ColorHistogram());

    // Load all training images to build universal vocabulary
    std::vector<cv::Mat> trainingImages = load_images(argv[1]);
    std::map<cv::Mat *, std::vector<std::vector<double>>> featuresForImages;

    std::cout << trainingImages.size() << std::endl;

    // Extract HoG features and Color Histograms using early fusion
    for(cv::Mat &img : trainingImages)
    {
        std::vector<std::vector<double>> imgFeatures;
        for(int j = 0; j < img.rows - regionSize.height + 1; j+=regionSize.height)
        {
            for(int i = 0; i <  img.cols - regionSize.width + 1; i+=regionSize.width)
            {
                cv::Mat region = img(cv::Rect(i, j, regionSize.width, regionSize.height));

                std::vector<double> featureVector;
                for(HistogramFeature *feat : features)
                {
                    std::vector<double> f = feat->Compute(region);
                    featureVector.insert(featureVector.end(), f.begin(), f.end());
                }

                imgFeatures.push_back(featureVector);
            }
        }

        featuresForImages[&img] = imgFeatures;
    }

    for(HistogramFeature *feat : features)
    {
        delete feat;
    }

    //Create Vocabulary Tree
    std::vector<std::vector<double>> featureSet;
    for(auto &feat : featuresForImages)
    {
        for(auto fv : feat.second)
            featureSet.push_back(fv);
    }

    double start = clock();
    std::cout << "Build Vocabulary Tree for " << featureSet.size() << " features" << std::endl;
    vocabulary_tree tree; //(4^4) = 256 words
    tree.K = 4; //branching factor
    tree.L = 4; //depth
    hierarchical_kmeans(featureSet, tree);
    std::cout << double( clock() - start ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;

    SaveVocabularyTree("universal_vocabulary.out", tree);
    return 0;
}
