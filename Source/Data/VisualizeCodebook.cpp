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

cv::Mat combineRegions(std::vector<cv::Mat> &regions){

    int rows = regions.size()/10;
    int cols = 10;
    int last_row = regions.size()%10;

    cv::Mat combined((rows+1)*54, cols*54, CV_8UC3, cv::Scalar(0));

    int region_ct = 0;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            for(int k = 0; k < 54; k++){
                for(int l = 0; l < 54; l++){
                    combined.at<cv::Vec3b>(i*54+k, j*54+l) = regions[region_ct].at<cv::Vec3b>(k, l);
                }
            }
            region_ct++;
        }
    }

    for(int j = 0; j < last_row; j++){
        for(int k = 0; k < 54; k++){
            for(int l = 0; l < 54; l++){
                combined.at<cv::Vec3b>(rows*54+k, j*54+l) = regions[region_ct].at<cv::Vec3b>(k,l);
            }
        }
        region_ct++;
    }

    return combined;
}

int main(int argc, char **argv)
{
    // Feature set
    std::vector<HistogramFeature *> features;
    features.push_back(new HistogramOfOrientedGradients());
    features.push_back(new ColorHistogram());

    // Load input images
    std::vector<cv::Mat> trainingImages = load_images(argv[1]);
    std::map<cv::Mat *, std::vector<std::vector<double>>> featuresForImages;
    std::map<cv::Mat *, std::vector<cv::Mat>> regionsForImages;

    // Extract HoG features and Color Histograms using early fusion
    for(cv::Mat &img : trainingImages)
    {
        std::vector<std::vector<double>> imgFeatures;
        std::vector<cv::Mat> regions;
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
                regions.push_back(region);
            }
        }

        featuresForImages[&img] = imgFeatures;
        regionsForImages[&img] = regions;
    }

    for(HistogramFeature *feat : features)
    {
        delete feat;
    }

    vocabulary_tree tree;
    LoadVocabularyTree(argv[2], tree);

    // get labels features for each feature
    VocabularyTreeQuantization quant(tree);
    int vocabulary_size = quant.size();
    std::vector<double> word_counts(vocabulary_size);
    std::vector<std::vector<cv::Mat>> quantized_features(vocabulary_size);
    for(cv::Mat &img: trainingImages)
    {
        std::vector<std::vector<double>> img_features = featuresForImages[&img];
        std::vector<cv::Mat> img_regions = regionsForImages[&img];

        for(int i = 0; i < img_features.size(); i++){
            int label = quant.get_hierarchical_label(img_features[i],tree.root, tree.K);
            word_counts[label]++;
            quantized_features[label].push_back(img_regions[i]);
        }
    }

    for(int i = 0; i < vocabulary_size; i++){
        std::cout << i << ", " << word_counts[i] << std::endl;
        cv::Mat examples = combineRegions(quantized_features[i]);

        std::ostringstream convert;
        //assumes the subfolder already exists
        convert << "vocabulary/word" << i << ".jpg";
        std::string s = convert.str();
        cv::imwrite(s, examples);
    }

    return 0;
}
