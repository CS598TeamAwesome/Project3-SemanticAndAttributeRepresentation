#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

int main(int argc, char **argv){
    std::ifstream labels(argv[1]); //results file

    int pos = 50; //assume correct labels for first N lines is +1
    int neg = 50; //assume correct labels for next N lines is -1

    double true_positive = 0;
    double false_positive = 0; //also called false alarm
    double true_negative = 0;
    double false_negative = 0;

    for(int i = 0; i < pos; i++){
        std::string s;
        std::getline(labels, s);

        if(s.compare("1") == 0){
            true_positive++;
        } else if(s.compare("-1") == 0){
            false_negative++;
        } else {
            std::cout << "unexpected label" << std::endl;
            //TODO: exception handling
        }
    }

    for(int i = 0; i < neg; i++){
        std::string s;
        std::getline(labels, s);

        if(s.compare("1") == 0){
            false_positive++;
        } else if(s.compare("-1") == 0){
            true_negative++;
        } else {
            std::cout << "unexpected label" << std::endl;
            //TODO: exception handling
        }
    }

    double sensitivity = true_positive /(true_positive + false_negative); //equivalent to recall
    double specificity = true_negative / (false_positive + true_negative); //true negative rate
    double precision = true_positive / (true_positive + false_positive);
    double npv = true_negative / (true_negative + false_negative);
    double accuracy = (true_positive + true_negative) / (pos + neg);

    std::cout << true_positive << ", " << false_negative << ", " << false_positive << ", " << true_negative << std::endl;
    std::cout <<sensitivity << ", " << specificity << ", " << precision << ", " << npv << ", " << accuracy << std::endl;
    return 0;
}
