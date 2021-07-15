#include <stdio.h>
#include <chrono>
#include <torch/script.h>
#include <iostream>
#include <memory> 

#include <opencv2/opencv.hpp>

#include "GP.h"
#include "weights.h"

int main(int argc, char* argv[]){

torch::jit::script::Module module;
try {
    module = torch::jit::load(argv[1]); 
}
catch (const c10::Error& e) {
    std::cerr << "error loading the model\n"; 
    return -1;
}
std::cout<< "ok\n"; 
}
