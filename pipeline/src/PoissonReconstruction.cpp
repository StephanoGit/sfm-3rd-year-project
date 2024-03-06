#include "../include/PoissonReconstruction.h"
#include "../include/CommonUtil.h"

bool PoissonReconstruction::generate_mesh(const std::string file_in,
                                          const std::string file_out) {
    std::cout << "===========================================" << std::endl;
    std::cout << "          Poisson Reconstruction           " << std::endl;
    std::cout << "===========================================" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(file_in, *cloud);

    if (cloud->size() <= 0) {
        std::cout << "ERROR: Failed to load pointcloud -- pcd file empty [❌]"
                  << std::endl;
        return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PolygonMesh mesh;

    // filter pointcloud -- remove points where x NOT ∈ [0.003, 0.83]
    pcl::PassThrough<pcl::PointXYZ> pass_through;
    pass_through.setInputCloud(cloud);
    pass_through.setFilterFieldName("x");
    pass_through.setFilterLimits(0.003, 0.83);
    pass_through.filter(*filtered_cloud);

    // remove outliers -- must have at least 150 neighbours in a 0.07 radius
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius_outlier_removal;
    radius_outlier_removal.setInputCloud(cloud);
    radius_outlier_removal.setRadiusSearch(0.07);
    radius_outlier_removal.setMinNeighborsInRadius(150);
    radius_outlier_removal.filter(*filtered_cloud);

    // create mesh
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setNumberOfThreads(8);
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setKSearch(10); // 20
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
        new pcl::PointCloud<pcl::Normal>());
    ne.compute(*cloud_normals);

    for (std::size_t i = 0; i < cloud_normals->size(); ++i) {
        cloud_normals->points[i].normal_x *= -1;
        cloud_normals->points[i].normal_y *= -1;
        cloud_normals->points[i].normal_z *= -1;
    }
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_smoothed_normals(
        new pcl::PointCloud<pcl::PointNormal>());
    pcl::concatenateFields(*cloud, *cloud_normals,
                           *cloud_smoothed_normals); // x

    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setDepth(9); // 9 // 7
    poisson.setInputCloud(cloud_smoothed_normals);
    poisson.setPointWeight(4);      // 4
    poisson.setSamplesPerNode(1.5); // 1.5
    poisson.setScale(1.1);          // 1.1
    poisson.setIsoDivide(8);        // 8
    poisson.setConfidence(1);
    poisson.setManifold(0);
    poisson.setOutputPolygons(0);
    poisson.setSolverDivide(8); // 8
    poisson.reconstruct(mesh);

    pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromPCLPointCloud2(mesh.cloud, *mesh_cloud);

    // Step 3: Associate colors with mesh vertices using k-d tree
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::io::loadPCDFile(file_in, *cloudRGB);
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloudRGB);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr mesh_cloud_colored(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    for (size_t i = 0; i < mesh_cloud->points.size(); ++i) {
        pcl::PointXYZRGB searchPoint;
        searchPoint.x = mesh_cloud->points[i].x;
        searchPoint.y = mesh_cloud->points[i].y;
        searchPoint.z = mesh_cloud->points[i].z;

        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        if (kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch,
                                  pointNKNSquaredDistance) > 0) {
            mesh_cloud_colored->points.push_back(
                cloudRGB->points[pointIdxNKNSearch[0]]);
        }
    }

    // Step 4: Update mesh with colors
    pcl::toPCLPointCloud2(*mesh_cloud_colored, mesh.cloud);
    pcl::io::savePLYFile(
        "../reconstructions/" + file_out + "/mesh/" + "rgb_mesh.ply", mesh);

    // visualize mesh
    display_mesh(mesh);

    display_point_cloud(mesh_cloud_colored);

    return true;
}
