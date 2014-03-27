#include <Feature/ColorHistogram.hpp>
#include <Feature/HistogramOfOrientedGradients.hpp>
#include <BagOfFeatures/Codewords.hpp>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace ColorTextureShape;
using namespace LocalDescriptorAndBagOfFeature;

cv::Size regionSize(54, 54); // Default HoG is 3x3 cell blocks of 6x6 pixel cells, this gives 3x3 block regions

std::vector<cv::Mat> load_images(std::string imageFileList)
{
    
}

int main(int argc, char **argv)
{
    // Feature set
    std::vector<HistogramFeature *> features =  { new HistogramOfOrientedGradients(), new ColorHistogram() };
    
    // Load input images
    std::vector<cv::Mat> images = load_image(argv[1]);
    std::vector<std::vector<double>> features;
    
    // Extract HoG features and Color Histograms using early fusion
    for(cv::Mat img : images)
    {
        for(int j = 0; j < img.rows - regionSize.height + 1; j++)
        {
            for(int i = 0; i <  img.cols - regionSize.width + 1; i++)
            {
                cv::Mat region = img(cv::Rect(i, j, regionSize.width, regionSize.height));
                
                std::vector<double> featureVector;                
                for(HistogramFeature *feat : features)
                {
                    std::vector<double> f = feat->Compute(region);
                    featureVector.insert(featureVector.end(), f);
                }
                
                features.push_back(featureVector);             
            }
        }
    }
    
    for(HistogramFeature *feat : features)
    {
        delete feat;
    }
    
    // Create codebook
    std::vector<std::vector<double>> codebook;
    FindCodewords(features, 100, codebook);
    
    // Save codebook
    SaveCodebook("codebook", codebook);
}
