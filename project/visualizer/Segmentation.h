#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/region_growing_rgb.h>

class Segmentation {

private:
public:
    Segmentation() {}
    ~Segmentation() {}
    void color_based_growing_segmentation();
};
