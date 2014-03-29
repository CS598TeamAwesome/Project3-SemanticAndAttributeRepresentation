#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

int main(int argc, char **argv){
    std::string command1 = "svm-train -v 5 -w1 ";
    std::string command2 = " -g ";
    std::string command3 = " scaled_svm_train_Africans";

    //std::array<double, 9> gamma = {-13, -11, -9, -7, -5, -3, -1, 1, 3};
    //std::array<double, 11> C = {-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15};

    double gamma_start = -13;
    double gamma_end = 3;
    double gamma_increment = 2;

    double C_start = -5;
    double C_end = 15;
    double C_increment = 2;

    std::ofstream outfile("coarse_grid_search.bat");
    std::string results = " >> coarse_grid_search_results.out";
/*
    double gamma_start = -4;
    double gamma_end = -3.5;
    double gamma_increment = 0.05;

    double C_start = .5;
    double C_end = 1;
    double C_increment = .05;

    std::ofstream outfile("fine_grid_search.bat");
    std::string results = " >> fine_grid_search_results.out";
*/
    double gamma_current = gamma_start;
    while(gamma_current <= gamma_end){
        double C_current = C_start;
        while(C_current <= C_end){
            std::ostringstream convert;
            convert << command1 << pow(2.0, C_current) << command2 << pow(2.0, gamma_current) << command3;
            std::string s = convert.str();
            outfile << "echo " << s << results << std::endl;
            outfile << s << results << std::endl;
            C_current += C_increment;
        }
        gamma_current += gamma_increment;
    }
    outfile.close();
}
