#ifndef _OCTREEMODEL_
#define _OCTREEMODEL

#include"iostream"
#include"fstream"
#include"Octree.h"
#include"vector"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>


class OctreeModel
{
public:
	typedef std::vector<std::vector<int>> Pixel;
	typedef std::vector<Pixel> Voxel;

	OctreeNode* Root;
	OctreeModel(OctreeNode* root,int dep = 7);

	void buildTree(OctreeNode* &root, int dep);

	Status judge(OctreeNode* &root);
	void getModel(OctreeNode* &node);
	void getSurface();
	void getNorm();
	void get_color();
	Eigen::Vector3f getNormal(int indX, int indY, int indZ);
	bool judgeSurface(int indexX, int indexY, int indexZ);

	void saveModel(const char* pFileName);
	void saveModelWithNormal(const char* pFileName);
	void saveModelWithColor(const char* pFileName);

	void loadMatrix(const char* pFileName);
	void loadImage(const char* pDir0, const char* pPrefix0, const char* pSuffix0, const char* pDir1, const char* pPrefix1, const char* pSuffix1);

protected:
	int depth;
	std::vector<Projection> m_projectionList;

	CoordinateInfo m_corrX;
	CoordinateInfo m_corrY;
	CoordinateInfo m_corrZ;

	Voxel voxel;
	std::vector<Point> surface;
	std::vector<Eigen::Vector3f> norm;
	std::vector<cv::Vec3i> surface_color;
	std::vector<Point> leaf;
};

#endif