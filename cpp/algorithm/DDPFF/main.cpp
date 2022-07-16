#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <rep/DDPFF.h>
#include "globals/Config.h"

void read_pcd(pointBuffer_t & points_ptr, const char * pcd_path) {
    std::ifstream infile(pcd_path);

    if (!infile.is_open()) {
        throw std::runtime_error("Input file not found!");
    }

    std::string line;
    // read header
    std::getline(infile, line);
    if (line != "ply") {
        throw std::runtime_error("Wrong input format: only PLY accepted, but '" + line + "' was provided!");
    }
    // skip header
    while (line != "end_header") {
        std::getline(infile, line);
    }

    // read data
    int i = 0;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        real_t x, y, z;
        iss >> x >> y >> z;
        Vec3 tmp;
        tmp << x, y, z;
        points_ptr[i] = tmp;
        i++;
    }
    std::cout << "Parsed point cloud with point: " << i << std::endl;
}

void save_planes(const std::vector<PlanePointNormal> & planes, const char * save_path) {
    std::ofstream output(save_path);

    for (const auto & plane : planes) {
        for (auto inlier : plane.inliers) {
            output << inlier << " ";
        }
        output << std::endl;
    }
}

void read_config(const char* cfg_path){
    std::ifstream input(cfg_path);

    std::string line;
    while (std::getline(input, line)){
        std::string param_name = line.substr(0, line.find('='));
        real_t value = std::stod(line.substr(line.find('=') + 1));
        if (param_name == "floodFill.pointThreshold_min") config.pointThresholdFloodFill_min = value;
        else if (param_name == "floodFill.pointThreshold_max") config.pointThresholdFloodFill_max = value;
        else if (param_name == "floodFill.planeThreshold_flood") config.planeThresholdFloodFill_flood = value;
        else if (param_name == "floodFill.planeThreshold_merge") config.planeThresholdFloodFill_merge = value;
        else if (param_name == "floodFill.planeThreshold_flood_max") config.planeThresholdFloodFill_flood_max = value;
        else if (param_name == "floodFill.planeThreshold_merge_max") config.planeThresholdFloodFill_merge_max = value;
        else if (param_name == "floodFill.angleThresholdFloodFill") config.angleThresholdFloodFill = value;
        else if (param_name == "floodFill.angleThresholdFloodFill_max") config.angleThresholdFloodFill_max = value;
        else if (param_name == "floodFill.minPlaneSize") config.minPlaneSize = value;
        else if (param_name == "floodFill.normalSampleDistance_min") config.normalSampleDistance_min = value;
        else if (param_name == "floodFill.normalSampleDistance_max") config.normalSampleDistance_max = value;
        else if (param_name == "floodFill.c_plane") config.c_plane = value;
        else if (param_name == "floodFill.c_plane_merge") config.c_plane_merge = value;
        else if (param_name == "floodFill.c_point") config.c_point = value;
        else if (param_name == "floodFill.c_angle") config.c_angle = value;
        else if (param_name == "floodFill.c_range") config.c_range = value;
    }   
}

int main(int argc, char** argv) {
    // arg1 -- pcd_path, arg2 -- config_path, arg3 -- output_path
    const char * pcd_path = argv[1];
    const char * config_path = argv[2];
    const char * output_path = argv[3];

    read_config(config_path);
   
    auto ddpff = new DDPFF();
    ddpff->init();

    auto * points = new pointBuffer_t();
    auto * colors = new colorBuffer_t();
    auto * depth = new depthBuffer_t();
    read_pcd(*points, pcd_path);
    ddpff->setBuffers(points, colors, depth);

    ddpff->compute();
    const std::vector<PlanePointNormal> & result = ddpff->getPlanes();
    for (const auto& plane : result) {
        std::cout << "Plane with " << plane.count << " points detected!" << std::endl;
    }
    std::cout << "Totally detected " << result.size() << " planes" << std::endl;
    save_planes(result, output_path);
    return 0;
}
