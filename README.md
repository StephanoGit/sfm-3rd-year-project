![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)

![GitHub last commit](https://img.shields.io/github/last-commit/StephanoGit/sfm-3rd-year-project)
![GitHub repo size](https://img.shields.io/github/repo-size/StephanoGit/sfm-3rd-year-project)

# Incremental Structure from Motion (SfM)
This repository presents one approach to the incremental Structure from Motion (SfM) pipeline, focusing on the reconstruction of 3D structures from sequences of images. This implementation prioritizes the processing of image pairs based on the homography inliers ratio, which can lead to more robust initial reconstructions and potentially improve the overall accuracy and efficiency of the pipeline.

![Pipeline](https://github.com/StephanoGit/sfm-3rd-year-project/blob/main/images/Pipeline.jpg)

## Installation (MACOS)
Before cloning the repository by using the following command:
```
git clone https://github.com/StephanoGit/sfm-3rd-year-project.git
```

make sure you have the following libraries installed:
```
brew install cmake opencv boost pcl ceres-solver
```


In the project directory, create the `build` directory and execute the following commands:
```
mkdir build
cd build
cmake ..
make
./sfm -c <default/iphone>
      -i <images/video>
      -s <resize>
      -d <path/to/images/folder>
      -x <SIFT/SURF/BRISK/KAZE/AKAZE/ORB>
      -m <FLANN/BF>
```
> ⚠ Warning ⚠
>
> - The `CMake` is machine-dependent. You should create your own.
> - You must provide your own camera parameters if you are using your own photos. These can be modified in the main.cpp file by hardcoding them or by using the `import_intrinsics(directory_K, K, directory_d, d)` function. For more infomration please check `IOUtil.h`
> - The `resize` value must be an `int`


## Future Improvements
This implementation offers a foundational approach to incremental structure from motion, showcasing commendable performance in exterior reconstructions, particularly with building facades. However, it exhibits limitations in handling dense feature environments typically encountered in interior reconstructions, occasionally leading to challenges in achieving successful reconstruction outcomes. This sensitivity arises from the algorithm's struggle to efficiently process an abundance of features, which can hinder the bundle adjustment's ability to reach convergence.

To enhance the robustness and versatility of this implementation, the following enhancements are proposed:
1. Integration of feature tracking through optical flow techniques, which promises improved accuracy and stability in feature detection and matching across successive frames. This advancement could significantly bolster the algorithm's efficiency in complex environments.
2. Adoption of more resilient bundle adjustment strategies, employing robust estimation techniques such as Huber, Cauchy, or Graduated Non-Convexity (GNC). These methodologies aim to improve the algorithm's tolerance to outliers and non-linearities, facilitating more reliable convergence even in challenging scenarios.


## References
```
@Book{Hartley2004,
    author = "Hartley, R.~I. and Zisserman, A.",
    title = "Multiple View Geometry in Computer Vision",
    edition = "Second",
    year = "2004",
    publisher = "Cambridge University Press, ISBN: 0521540518"
}

@article{snavely2008modeling, 
    title={Modeling the world from internet photo collections}, 
    author={Snavely, Noah and Seitz, Steven M and Szeliski, Richard}, 
    journal={International journal of computer vision}, 
    volume={80}, 
    pages={189--210}, 
    year={2008}, 
    publisher={Springer}
}

@misc{royshil,
  author = {royshil},
  title = {SfM-Toy-Library},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/royshil/SfM-Toy-Library}}
}
<<<<<<< HEAD
```
=======
```
>>>>>>> b9ba70e (remove large file such as videos)
