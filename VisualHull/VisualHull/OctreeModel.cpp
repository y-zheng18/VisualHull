#pragma warning(disable:4819)
#include"iostream"
#include"fstream"
#include"OctreeModel.h"
#include"vector"
#include"queue"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

OctreeModel::OctreeModel(OctreeNode* root, int dep) : Root(root), depth(dep),
m_corrX(root->xMax, -5, 5), m_corrY(root->yMax, -10, 10), m_corrZ(root->zMax, 15, 30)
{
	voxel = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<int>(m_corrZ.m_resolution, 1)));
}

void OctreeModel::loadMatrix(const char* pFileName)//加载所有projection
{
	std::ifstream fin(pFileName);

	int num;
	Eigen::Matrix<float, 3, 3> matInt;
	Eigen::Matrix<float, 3, 4> matExt;
	Projection projection;
	while (fin >> num)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				fin >> matInt(i, j);

		double temp;
		fin >> temp;
		fin >> temp;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				fin >> matExt(i, j);

		projection.m_projMat = matInt * matExt;//相机参数
		m_projectionList.push_back(projection);
	}
}

void OctreeModel::loadImage(const char* pDir0, const char* pPrefix0, const char* pSuffix0, const char* pDir1, const char* pPrefix1, const char* pSuffix1)//对每个projection输入image
{
	int fileCount = m_projectionList.size();
	std::string fileName0(pDir0);
	fileName0 += '/';
	fileName0 += pPrefix0;

	std::string fileName1(pDir1);
	fileName1 += '/';
	fileName1 += pPrefix1;

	for (int i = 0; i < fileCount; i++)
	{
		std::cout << fileName0 + std::to_string(i) + pSuffix0 << std::endl;
		m_projectionList[i].m_image = cv::imread(fileName0 + std::to_string(i) + pSuffix0, CV_8UC1);
		std::cout << fileName1 + std::to_string(i) + pSuffix1 << std::endl;
		m_projectionList[i].m_color = cv::imread(fileName1 + std::to_string(i) + pSuffix1);
	}
}

void OctreeModel::buildTree(OctreeNode* &root, int dep)
{
	if (dep > depth)
		return;

	if (dep <= 4)															//开始时递归到5层，不进行方块位置判断
	{
		int dx = (root->xMax - root->xMin) / 2;
		int dy = (root->yMax - root->yMin) / 2;
		int dz = (root->zMax - root->zMin) / 2;
		root->child[0] = new OctreeNode(ON, root->xMin, root->xMin + dx, root->yMin, root->yMin + dy, root->zMin, root->zMin + dz);//左下后
		root->child[1] = new OctreeNode(ON, root->xMin + dx, root->xMax, root->yMin, root->yMin + dy, root->zMin, root->zMin + dz);//左下前
		root->child[2] = new OctreeNode(ON, root->xMin, root->xMin + dx, root->yMin + dy, root->yMax, root->zMin, root->zMin + dz);//右下后
		root->child[3] = new OctreeNode(ON, root->xMin + dx, root->xMax, root->yMin + dy, root->yMax, root->zMin, root->zMin + dz);//右下前
		root->child[4] = new OctreeNode(ON, root->xMin, root->xMin + dx, root->yMin, root->yMin + dy, root->zMin + dz, root->zMax);//左上后
		root->child[5] = new OctreeNode(ON, root->xMin + dx, root->xMax, root->yMin, root->yMin + dy, root->zMin + dz, root->zMax);//左上前
		root->child[6] = new OctreeNode(ON, root->xMin, root->xMin + dx, root->yMin + dy, root->yMax, root->zMin + dz, root->zMax);//右上后
		root->child[7] = new OctreeNode(ON, root->xMin + dx, root->xMax, root->yMin + dy, root->yMax, root->zMin + dz, root->zMax);//右上前
		for (int i = 0; i < 8; i++)
		{
			buildTree(root->child[i], dep + 1);
		}
	}
	else if (dep < depth)
	{
		root->status = judge(root);							//判断方块位置，若全为IN或OUT则无需递归，只有ON时需要继续递归
		if (root->status != ON)
		{
			getModel(root);
			return;
		}
		int dx = (root->xMax - root->xMin) / 2;
		int dy = (root->yMax - root->yMin) / 2;
		int dz = (root->zMax - root->zMin) / 2;
		root->child[0] = new OctreeNode(ON, root->xMin, root->xMin + dx, root->yMin, root->yMin + dy, root->zMin, root->zMin + dz);//左下后
		root->child[1] = new OctreeNode(ON, root->xMin + dx, root->xMax, root->yMin, root->yMin + dy, root->zMin, root->zMin + dz);//左下前
		root->child[2] = new OctreeNode(ON, root->xMin, root->xMin + dx, root->yMin + dy, root->yMax, root->zMin, root->zMin + dz);//右下后
		root->child[3] = new OctreeNode(ON, root->xMin + dx, root->xMax, root->yMin + dy, root->yMax, root->zMin, root->zMin + dz);//右下前
		root->child[4] = new OctreeNode(ON, root->xMin, root->xMin + dx, root->yMin, root->yMin + dy, root->zMin + dz, root->zMax);//左上后
		root->child[5] = new OctreeNode(ON, root->xMin + dx, root->xMax, root->yMin, root->yMin + dy, root->zMin + dz, root->zMax);//左上前
		root->child[6] = new OctreeNode(ON, root->xMin, root->xMin + dx, root->yMin + dy, root->yMax, root->zMin + dz, root->zMax);//右上后
		root->child[7] = new OctreeNode(ON, root->xMin + dx, root->xMax, root->yMin + dy, root->yMax, root->zMin + dz, root->zMax);//右上前
		for (int i = 0; i < 8; i++)
		{
			buildTree(root->child[i], dep + 1);

		}
	}
	else if (dep == depth)//叶子节点
	{
		//std::cout << root->xMin << " " << root->xMax << "   ";
		Point P(root->xMin, root->yMin, root->zMin);
		leaf.push_back(P);
		int size = m_projectionList.size();
		for (int x = root->xMin; x < root->xMax; x++)
			for (int y = root->yMin; y < root->yMax; y++)
				for (int z = root->zMin; z < root->zMax; z++)
				{
					int flag = 1;
					double cX = m_corrX.index2coor(x);
					double cY = m_corrY.index2coor(y);
					double cZ = m_corrZ.index2coor(z);
					for (int j = 0; j < size; j++)
					{
						flag = m_projectionList[j].checkRange(cX, cY, cZ);
						if (!flag)
							break;
					}
					voxel[x][y][z] = flag;
				}
	}
}

Status OctreeModel::judge(OctreeNode* &root)	
{
	int sum = 0;
	int X_index[8] = { root->xMin, root->xMin, root->xMin, root->xMin,root->xMax, root->xMax,root->xMax, root->xMax };
	int Y_index[8] = { root->yMin, root->yMax, root->yMin, root->yMax,root->yMin, root->yMax,root->yMin, root->yMax };
	int Z_index[8] = { root->zMin, root->zMin, root->zMax, root->zMax,root->zMin, root->zMin,root->zMax, root->zMax };

	int size = m_projectionList.size();
	for (int i = 0; i < 8; i++)
	{
		int flag = 1;
		double cX = m_corrX.index2coor(X_index[i]);
		double cY = m_corrY.index2coor(Y_index[i]);
		double cZ = m_corrZ.index2coor(Z_index[i]);
		for (int j = 0; j < size; j++)
		{
			flag = m_projectionList[j].checkRange(cX, cY, cZ);
			if (!flag)
				break;
		}
		sum += flag;
	}
	if (sum == 8)
	{
		return IN;
	}
	if (sum > 0)
	{
		return ON;
	}
	
	//若八个顶点都为out
	int cX_index[6] = { root->xMin, root->xMin, (root->xMin + root->xMax) / 2, (root->xMin + root->xMax) / 2,root->xMin, root->xMax};
	int cY_index[6] = { root->yMin, root->yMax,	root->yMin, root->yMax, (root->yMin + root->yMax) / 2, (root->yMax,root->yMin + root->yMax) / 2};
	int cZ_index[6] = { (root->zMin + root->zMax) / 2, (root->zMin + root->zMax) / 2, root->zMax, root->zMax,root->zMin, root->zMin};
	//判断棱心位置以及中心位置
	int flag = 1;
	for (int k = 0; k < 6; k++)
	{
		double cX = m_corrX.index2coor(cX_index[k]);
		double cY = m_corrY.index2coor(cY_index[k]);
		double cZ = m_corrZ.index2coor(cZ_index[k]);
		for (int s = 0; s < size; s++)
		{
			flag = m_projectionList[s].checkRange(cX, cY, cZ);
			if (!flag)
				break;
		}
		if (flag)
		{
			return ON;
		}
	}
	return OUT;
}


void OctreeModel::getModel(OctreeNode* &node)
{
	for (int i = node->xMin; i < node->xMax; i++)
		for (int j = node->yMin; j < node->yMax; j++)
			for (int t = node->zMin; t < node->zMax; t++)
			{
				voxel[i][j][t] = (int)node->status;
			}
}

bool OctreeModel::judgeSurface(int indexX, int indexY, int indexZ)
{
	// lambda表达式，用于判断某个点是否在Voxel的范围内
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	// 邻域：上、下、左、右、前、后。
	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };

	if (voxel[indexX][indexY][indexZ] != 1)
		return false;
	bool ans = false;
	for (int k = 0; k < 6; k++)
	{
		ans = ans || outOfRange(indexX + dx[k], indexY + dy[k], indexZ + dz[k])
			|| !voxel[indexX + dx[k]][indexY + dy[k]][indexZ + dz[k]];
	}
	if (ans)
	{
		voxel[indexX][indexY][indexZ] = 2;
		return true;
	}
	return false;
}

void OctreeModel::getSurface()
{
	// lambda表达式，用于判断某个点是否在Voxel的范围内
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	// 邻域：上、下、左、右、前、后。
	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };

	std::queue<Point> surfaceList;							//bfs广度优先搜索，从一个点开始遍历全部表面点
	for (int x = leaf[0].x; x <= leaf[0].x + 4; x++)
		for (int y = leaf[0].y; y <= leaf[0].y + 4; y++)
			for (int z = leaf[0].z; z <= leaf[0].z + 4; z++)
			{
				if (judgeSurface(x, y, z))
				{
					Point p0(x, y, z);
					surfaceList.push(p0);
					surface.push_back(p0);
				}
			}

	while (!surfaceList.empty())
	{
		Point p1 = surfaceList.front();
		surfaceList.pop();
		for (int dx = -1; dx <= 1; dx++)
			for (int dy = -1; dy <= 1; dy++)
				for (int dz = -1; dz <= 1; dz++)
				{
					if (judgeSurface(p1.x + dx, p1.y + dy, p1.z + dz))
					{
						Point neighbor(p1.x + dx, p1.y + dy, p1.z + dz);
						surfaceList.push(neighbor);
						surface.push_back(neighbor);
					}
				}
	}
}

void OctreeModel::get_color()													//获取颜色
{
	
	int size = m_projectionList.size();
	for (int i = 0; i < surface.size(); i++)
	{
		cv::Vec3i ave_color = cv::Vec3b(0, 0, 0);
		cv::Vec3i color = cv::Vec3b(0, 0, 0);
		int indexX = surface[i].x;
		int indexY = surface[i].y;
		int indexZ = surface[i].z;
		double coorX = m_corrX.index2coor(indexX);
		double coorY = m_corrY.index2coor(indexY);
		double coorZ = m_corrZ.index2coor(indexZ);
		for (int j = 0; j < size; j++)
		{
			ave_color += m_projectionList[j].get_color(coorX, coorY, coorZ);
			
		}
		ave_color(0) = ave_color(0) / size;
		ave_color(1) = ave_color(1) / size;
		ave_color(2) = ave_color(2) / size;										//取平均值
		surface_color.push_back(ave_color);
	}
}

void OctreeModel::getNorm()														//获取表面点云对应的法向量
{
	for (int i = 0; i < surface.size(); i++)
	{
		int indexX = surface[i].x;
		int indexY = surface[i].y;
		int indexZ = surface[i].z;
		double coorX = m_corrX.index2coor(indexX);
		double coorY = m_corrY.index2coor(indexY);
		double coorZ = m_corrZ.index2coor(indexZ);
		Eigen::Vector3f nor = getNormal(indexX, indexY, indexZ);
		norm.push_back(nor);
	}
}

void OctreeModel::saveModel(const char* pFileName)								//输出表面点
{
	std::ofstream fout(pFileName);
	FILE* file = fopen(pFileName, "w");

	for (int i = 0; i < surface.size(); i++)
	{
		int indexX = surface[i].x;
		int indexY = surface[i].y;
		int indexZ = surface[i].z;
		double coorX = m_corrX.index2coor(indexX);
		double coorY = m_corrY.index2coor(indexY);
		double coorZ = m_corrZ.index2coor(indexZ);
		fprintf(file, "%f %f %f\n", coorX, coorY, coorZ);
	}
	fclose(file);
}

void OctreeModel::saveModelWithNormal(const char* pFileName) //without color
{
	FILE* file = fopen(pFileName, "w");//without color
	for (int i = 0; i < surface.size(); i++)
	{
		int indexX = surface[i].x;
		int indexY = surface[i].y;
		int indexZ = surface[i].z;
		double coorX = m_corrX.index2coor(indexX);
		double coorY = m_corrY.index2coor(indexY);
		double coorZ = m_corrZ.index2coor(indexZ);
		fprintf(file, "%f %f %f %f %f %f\n", coorX, coorY, coorZ, norm[i](0), norm[i](1), norm[i](2));//without color
	}
	fclose(file);
}

void OctreeModel::saveModelWithColor(const char* pFileName) //with color
{
	FILE* file = fopen(pFileName, "w");
	fprintf(file, "%s\n", "ply");
	fprintf(file, "%s\n", "format ascii 1.0");
	fprintf(file, "%s%d\n", "element vertex ", surface.size());
	fprintf(file, "%s\n", "property float x");
	fprintf(file, "%s\n", "property float y");
	fprintf(file, "%s\n", "property float z");
	fprintf(file, "%s\n", "property float nx");
	fprintf(file, "%s\n", "property float ny");
	fprintf(file, "%s\n", "property float nz");
	fprintf(file, "%s\n", "property uchar red");
	fprintf(file, "%s\n", "property uchar green");
	fprintf(file, "%s\n", "property uchar blue");
	fprintf(file, "%s\n", "end_header");					
	//输入header
	for (int i = 0; i < surface.size(); i++)
	{
		int indexX = surface[i].x;
		int indexY = surface[i].y;
		int indexZ = surface[i].z;
		double coorX = m_corrX.index2coor(indexX);
		double coorY = m_corrY.index2coor(indexY);
		double coorZ = m_corrZ.index2coor(indexZ);
		fprintf(file, "%g %g %g %g %g %g %d %d %d\n", 
			coorX, coorY, coorZ, 
			norm[i](0), norm[i](1), norm[i](2), 
			surface_color[i](2), surface_color[i](1), surface_color[i](0));
	}
	fclose(file);
}

Eigen::Vector3f OctreeModel::getNormal(int indX, int indY, int indZ)
{
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	std::vector<Eigen::Vector3f> neiborList;
	std::vector<Eigen::Vector3f> innerList;

	for (int dX = -3; dX <= 3; dX++)
		for (int dY = -3; dY <= 3; dY++)
			for (int dZ = -3; dZ <= 3; dZ++)
			{
				if (!dX && !dY && !dZ)
					continue;
				int neiborX = indX + dX;
				int neiborY = indY + dY;
				int neiborZ = indZ + dZ;
				if (!outOfRange(neiborX, neiborY, neiborZ))
				{
					double coorX = m_corrX.index2coor(neiborX);
					double coorY = m_corrY.index2coor(neiborY);
					double coorZ = m_corrZ.index2coor(neiborZ);
					if (voxel[neiborX][neiborY][neiborZ] == 2)
						neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));//-----------------------------------
					else if (voxel[neiborX][neiborY][neiborZ] == 1)
						innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));//找到内部点和表面点，挖去内部点？
				}
			}

	Eigen::Vector3f point(m_corrX.index2coor(indX), m_corrY.index2coor(indY), m_corrZ.index2coor(indZ));

	Eigen::MatrixXf matA(3, neiborList.size());
	for (int i = 0; i < neiborList.size(); i++)
		matA.col(i) = neiborList[i] - point;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);

	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	for (auto const& vec : innerList)
		innerCenter += vec;
	innerCenter /= innerList.size();

	if (normalVector.dot(point - innerCenter) < 0)
		normalVector *= -1;
	return normalVector;
}