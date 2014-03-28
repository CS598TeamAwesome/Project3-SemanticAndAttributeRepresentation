#include <Feature/HistogramOfOrientedGradients.hpp>
#include <Feature/ColorHistogram.hpp>
#include <BagOfFeatures/Codewords.hpp>
#include <Quantization/HardAssignment.hpp>
#include <Quantization/VocabularyTreeQuantization.hpp>
#include <Util/Clustering.hpp>
#include <vector>
#include <fstream>
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

    // Load input images
    std::vector<cv::Mat> posImages = load_images(argv[1]);  
    std::vector<cv::Mat> negImages = load_images(argv[2]);
    std::map<cv::Mat *, std::vector<std::vector<double>>> featuresForImages;

    // Extract HoG features and Color Histograms using early fusion
    for(cv::Mat &img : posImages)
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
    
    for(cv::Mat &img : negImages)
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

/*
    // Load codebook
    std::vector<std::vector<double>> codebook;
    LoadCodebook(argv[3], codebook);
*/
    vocabulary_tree tree;
    LoadVocabularyTree("tree", tree);

    // Compute bag of features for each testing image
    //HardAssignment quant(codebook);
    VocabularyTreeQuantization quant(tree);
    std::map<cv::Mat *, std::vector<double>> testingBoW;
    for(cv::Mat &img: posImages)
    {
        std::vector<double> bow;
        quant.quantize(featuresForImages[&img], bow);
        
        testingBoW[&img] = bow;
    }
    
    for(cv::Mat &img: negImages)
    {
        std::vector<double> bow;
        quant.quantize(featuresForImages[&img], bow);
        
        testingBoW[&img] = bow;
    }

    // Save the training file in LibSVM format
    std::ofstream testingFile("test.out");
    for(cv::Mat &img : posImages)
    {
        testingFile << "+1 ";
        
        for(int j = 0; j < testingBoW[&img].size(); j++)
        {
            if(testingBoW[&img][j] != 0)
            {
                testingFile << j + 1 << ":" << testingBoW[&img][j] << " ";
            }
        }
        
        testingFile << std::endl;
    }
    
    for(cv::Mat &img : negImages)
    {
        testingFile << "-1 ";
        
        for(int j = 0; j < testingBoW[&img].size(); j++)
        {
            if(testingBoW[&img][j] != 0)
            {
                testingFile << j + 1 << ":" << testingBoW[&img][j] << " ";
            }
        }
        
        testingFile << std::endl;
    }
    testingFile.close();
    return 0;
}
