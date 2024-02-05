#ifndef __SEGMENTATION
#define __SEGMENTATION

#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/region_growing_rgb.h>

class Segmentation {
private:
public:
    Segmentation(){};
    ~Segmentation(){};

    bool run_segmentation(std::string file_path);
};

#endif
