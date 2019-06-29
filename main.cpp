#pragma warning(disable:4819)
#pragma warning(disable:4244)
#pragma warning(disable:4267)

#include"Octree.h"
#include"OctreeModel.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <limits>


int main(int argc, char** argv)
{
	clock_t t = clock(), start_t;

	start_t = clock();
	// 分别设置xyz方向的Voxel分辨率
	OctreeNode* Root = new OctreeNode;
	OctreeModel model(Root, 8);

	// 读取相机的内外参数
	model.loadMatrix("../../calibParamsI.txt");

	// 读取投影图片
	model.loadImage("../../wd_segmented", "WD2_", "_00020_segmented.png", "../../wd_data", "WD2_", "_00020.png");

	// 得到Voxel模型
	model.buildTree(Root, 0);
	model.getSurface();
	std::cout << "get model and surface done\n";
	std::cout << "time: " << (float(clock() - start_t) / CLOCKS_PER_SEC) << "seconds\n";

	start_t = clock();
	model.get_color();
	std::cout << "get color done\n";
	std::cout << "time: " << (float(clock() - start_t) / CLOCKS_PER_SEC) << "seconds\n";

	start_t = clock();
	// 将模型导出为xyz格式
	model.saveModel("../../WithoutNormal.xyz");
	std::cout << "save without normal done\n";
	std::cout << "time: " << (float(clock() - start_t) / CLOCKS_PER_SEC) << "seconds\n";

	start_t = clock();
	model.getNorm();
	model.saveModelWithNormal("../../WithNormal.xyz");
	std::cout << "save with normal done\n";
	std::cout << "time: " << (float(clock() - start_t) / CLOCKS_PER_SEC) << "seconds\n";

	start_t = clock();
	system("PoissonRecon.x64 --in ../../WithNormal.xyz --out ../../mesh.ply");
	std::cout << "save mesh.ply done\n";
	std::cout << "time: " << (float(clock() - start_t) / CLOCKS_PER_SEC) << "seconds\n";

	start_t = clock();
	model.saveModelWithColor("../../WithColor.ply");
	std::cout << "save with color and normal done\n";
	std::cout << "time: " << (float(clock() - start_t) / CLOCKS_PER_SEC) << "seconds\n";
	
	
	start_t = clock();
	system("PoissonRecon_color --in ../../WithColor.ply --out ../../mesh_color.ply --colors");
	std::cout << "save mesh_color.ply done\n";
	std::cout << "time: " << (float(clock() - start_t) / CLOCKS_PER_SEC) << "seconds\n";

	t = clock() - t;
	std::cout << "total time used: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	return (0);
}