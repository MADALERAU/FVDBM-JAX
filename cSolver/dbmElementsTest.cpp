#include <iostream>
#include "dbmElements.hh"
#include "Eigen/Dense"
#include <chrono>

using Eigen::Vector;
using Eigen:: Matrix;

int main(){
    Vector<double,9> testPDF {{1,2,2,2,2,3,3,3,3}};
    Vector<double,2> testVel {{0,0}};
    Cell testGE(testPDF,1);
    testGE.setEq();

    std::cout << "Neq: " << testGE.getNeq() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0;i<1000;i++){
        testGE.setEq();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    std::cout << "Execution time: " << duration.count() << "ms" <<std::endl;
}