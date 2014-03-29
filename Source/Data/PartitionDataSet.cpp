#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

const std::array<std::string, 10> wang_categories =
{
    "Africans",
    "Beaches",
    "Buildings",
    "Buses",
    "Dinosaurs",
    "Elephants",
    "Flowers",
    "Horses",
    "Mountains",
    "Food"
};

void produce_filelist();

int main(int argc, char **argv){
    produce_filelist();
}

void produce_filelist(){
    std::srand(time(0));

    std::ofstream complete_train("all_train");
    std::ofstream complete_test("all_test");
    for(int i = 0; i < 10; i++){
        std::ofstream train_file(wang_categories[i] + "_train");
        std::ofstream test_file(wang_categories[i] + "_test");

        std::string path_prefix = "wang_images/";

        //0-99 Africans, 100-199 Beaches, 200-299 Buildings, etc..
        std::array<int, 100> image_indexes;
        for(int j = 0; j < 100; j++){
            image_indexes[j] = i*100 + j;
        }

        //pick 50 random images within category to be train
        for(int i = 0; i < 50; i++){
            //pick random image in window from i to end
            int index = i + std::rand()%(100-i);

            //build image path and write to file
            std::ostringstream convert;
            convert << path_prefix << image_indexes[index] << ".jpg";
            std::string s = convert.str();

            train_file << s << std::endl;
            complete_train << s << std::endl;

            //swap chosen image into the current index to avoid collision (choosing the same image twice)
            std::swap(image_indexes[index], image_indexes[i]);
        }

        //the other 50 will be test
        for(int i = 50; i < 100; i++){
            //build image path and write to file
            std::ostringstream convert;
            convert << path_prefix << image_indexes[i] << ".jpg";
            std::string s = convert.str();

            test_file << s << std::endl;
            complete_test << s << std::endl;
        }

        train_file.close();
        test_file.close();
    }
    complete_train.close();
    complete_test.close();
}
