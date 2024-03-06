#ifndef __POISSON
#define __POISSON

#include "../include/SfmStructures.h"
#include <opencv2/core/mat.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/poisson.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

class PoissonReconstruction {
private:
public:
    PoissonReconstruction(){};
    ~PoissonReconstruction(){};

    static bool generate_mesh(const std::string file_in,
                              const std::string file_out);
};

#endif
