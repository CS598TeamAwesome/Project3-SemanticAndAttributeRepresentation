#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

#if !defined(WIN32) && !defined(WIN64)
#   define LINUX
#endif

int main(int argc, char **argv){
    std::string command1 = "svm-train -w1 ";
    std::string command2 = " -g ";
    std::string command3 = " " + std::string(argv[1]);

    double gamma_start = -100;
    double gamma_end = 100;
    double gamma_increment = 2;

    double C_start = -100;
    double C_end = 100;
    double C_increment = 2;
    
    std::string results = " >> coarse_grid_search_results.out";
    std::ofstream outfile;
    
    if(std::string(argv[3]) == "fine")
    {
        gamma_start = atof(argv[4]);
        gamma_end = atof(argv[5]);
        gamma_increment = 0.05;
    
        C_start = atof(argv[6]);
        C_end = atof(argv[7]);
        C_increment = .05;
    
#   ifdef LINUX   
        outfile.open("fine_grid_search");
        outfile << "#!/bin/sh" << std::endl;
#   else
        std::ofstream outfile("fine_grid_search.bat");
#   endif
        
        results = " >> fine_grid_search_results.out";
    }
    else
    {
#   ifdef LINUX   
        outfile.open("coarse_grid_search");
        outfile << "#!/bin/sh" << std::endl;
#   else
        std::ofstream outfile("coarse_grid_search.bat");
#   endif
    }
    
    double gamma_current = gamma_start;
    while(gamma_current <= gamma_end){
        double C_current = C_start;
        while(C_current <= C_end){
            std::ostringstream convert;
            convert << command1 << pow(2.0, C_current) << command2 << pow(2.0, gamma_current) << command3;
            std::string s = convert.str();
            outfile << "echo " << s << results << std::endl;
            outfile << s << results << std::endl;
            outfile << "svm-predict" << " " << argv[2] << command3 << ".model results.out" << results << std::endl;
            C_current += C_increment;
        }
        gamma_current += gamma_increment;
    }
    outfile.close();
}
