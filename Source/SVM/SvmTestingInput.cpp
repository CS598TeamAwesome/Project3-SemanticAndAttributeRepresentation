#include <Feature/HistogramOfOrientedGradients.hpp>
#include <Feature/ColorHistogram.hpp>
#include <BagOfFeatures/Codewords.hpp>
#include <Quantization/HardAssignment.hpp>
#include <vector>
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
        images.push_back(img);
    }

    return images;
}

int main(int argc, char **argv)
{
    // Feature set
    std::vector<HistogramFeature *> features = { new HistogramOfOrientedGradients(), new ColorHistogram() };

    // Load input images
    std::vector<cv::Mat> images = load_images(argv[1]);
    std::vector<std::vector<std::vector<double>>> featuresForImages;

    // Extract HoG features and Color Histograms using early fusion
    for(cv::Mat img : images)
    {
        std::vector<std::vector<double>> imgFeatures;
        for(int j = 0; j < img.rows - regionSize.height + 1; j++)
        {
            for(int i = 0; i <  img.cols - regionSize.width + 1; i++)
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

        featuresForImages.push_back(imgFeatures);
    }

    for(HistogramFeature *feat : features)
    {
        delete feat;
    }

    // Load codebook
    std::vector<std::vector<double>> codebook;
    LoadCodebook(argv[2], codebook);

    // Compute bag of features for each testing image
    HardAssignment quant(codebook);
    std::vector<std::vector<double>> testingBoW;
    for(int i = 0; i < images.size(); i++)
    {
        std::vector<double> bow;
        quant.quantize(featuresForImages[i], bow);

        testingBoW.push_back(bow);
    }

