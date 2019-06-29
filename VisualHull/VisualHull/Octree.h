#ifndef _OCTREE_H
#define _OCTREE_H
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
enum Status { OUT, IN, ON };
struct Point
{
	int x;
	int y;
	int z;
	Point(int x1 = 0,int y1 = 0, int z1 = 0) :x(x1), y(y1), z(z1) {}
};

struct OctreeNode
{
	Status status;
	int xMax, xMin;
	int yMax, yMin;
	int zMax, zMin;
	OctreeNode* child[8];									//约定child0->7为从左下角到右上角
	OctreeNode(Status sta = ON,
		int xmin = 0, int xmax = 512,
		int ymin = 0, int ymax = 512,
		int zmin = 0, int zmax = 512) :
		xMax(xmax), yMax(ymax), zMax(zmax),
		xMin(xmin), yMin(ymin), zMin(zmin),
		status(sta) {}

};

struct CoordinateInfo
{
	int m_resolution;
	double m_min;
	double m_max;

	double index2coor(int index)
	{
		return m_min + index * (m_max - m_min) / m_resolution;
	}

	CoordinateInfo(int resolution = 10, double min = 0.0, double max = 100.0)
		: m_resolution(resolution)
		, m_min(min)
		, m_max(max)
	{
	}
};

struct Projection
{
	Eigen::Matrix<float, 3, 4> m_projMat;
	cv::Mat m_image;
	cv::Mat m_color;
	const uint m_threshold = 125;

	bool outOfRange(int x, int max)
	{
		return x < 0 || x >= max;
	}

	bool checkRange(double x, double y, double z)
	{
		Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);
		int indX = vec3[1] / vec3[2];
		int indY = vec3[0] / vec3[2];

		if (outOfRange(indX, m_image.size().height) || outOfRange(indY, m_image.size().width))
			return false;
		return m_image.at<uchar>((uint)(vec3[1] / vec3[2]), (uint)(vec3[0] / vec3[2])) > m_threshold; //白色返回1
	}

	cv::Vec3i get_color(double x, double y, double z)
	{
		Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);
		int indX = vec3[1] / vec3[2];
		int indY = vec3[0] / vec3[2];
		return m_color.at<cv::Vec3b>((uint)(vec3[1] / vec3[2]), (uint)(vec3[0] / vec3[2]));
	}
};

#endif