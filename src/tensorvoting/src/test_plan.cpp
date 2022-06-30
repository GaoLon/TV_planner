// system level
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <memory>
#include <time.h>
#include <assert.h>
#include <vector>

// CUDA related
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <vector_types.h>

// ROS related
#include "ros/ros.h"
#include "ros/console.h"
#include "pointmatcher/PointMatcher.h"
#include "pointmatcher/IO.h"
#include "pointmatcher_ros/point_cloud.h"
#include "pointmatcher_ros/transform.h"
#include "get_params_from_server.h"
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>

// libpointmatcher
#include "aliases.h"

// libCudaVoting
#include "tensor_voting.h"
#include "CudaVoting.h"

// PCL related
#include <pcl/point_cloud.h>
#include <pcl/point_types.h> 
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#define USE_GPU_SPARSE_BALL
#define KNN 0
#define SHOTEST 1
#define TENSORVOTING 2

using namespace PointMatcherSupport;
using namespace std;
using namespace topomap;
using namespace cudavoting;

class VotingPlanner
{
    ros::NodeHandle & n;
    // subscriber
	ros::Subscriber cloudSub, wayPointSub;
	string cloudTopicIn;
    // publisher
	ros::Publisher cloudPub, pathPub;
	string cloudTopicOut;

	DP cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr kdcloud;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    unsigned int numPoints;
    int KNN_K;
    PointMatcher<float>::Matrix stick;
    PointMatcher<float>::Matrix plate;
    PointMatcher<float>::Matrix ball;
    PointMatcher<float>::Matrix normals;
    PointMatcher<float>::Matrix optdirection;
    std::vector<Eigen::Vector3d> path;
    bool get_cloud;
    Eigen::Vector3f start_pt;

    // tensor voting
    float sigma; // sparse voting kernel size

    // parameter
    bool savevtk; // whether save sequence of vtk files.
	std::string mapFrame;

public:
	VotingPlanner(ros::NodeHandle& n);
	void gotCloud(const sensor_msgs::PointCloud2ConstPtr cloudMsgIn);
	void waypointCallback(const geometry_msgs::PoseStampedConstPtr &msg);
    void drawPath(int type);
    void process(DP & cloud, float sigma);
    void dijkstra(size_t start, size_t end, int type);
    std::vector<Eigen::Vector3d> getPath() {return path;}
};

VotingPlanner::VotingPlanner(ros::NodeHandle& n):
    n(n), get_cloud(false), start_pt(0.0, 0.0, 1.0), KNN_K(10), \
    kdcloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>())
{
	// ROS initialization
    n.param("mapFrameId", mapFrame, std::string("map"));
    sigma = getParam<double>("sigma", 1.0);
	cloudTopicIn = getParam<string>("cloudTopicIn", "/point_cloud");
	cloudTopicOut = getParam<string>("cloudTopicOut", "/point_cloud_out");
    savevtk = getParam<bool>("savevtk", true);
    KNN_K = getParam<int>("KNN_K", 10);

	cloudSub = n.subscribe(cloudTopicIn, 100, &VotingPlanner::gotCloud, this);
	wayPointSub = n.subscribe("/move_base_simple/goal", 10, &VotingPlanner::waypointCallback, this);
    pathPub = n.advertise<visualization_msgs::Marker>(
		"/path", 1
    );
	cloudPub = n.advertise<sensor_msgs::PointCloud2>(
		getParam<string>("cloudTopicOut", "/point_cloud_sparsevoting"), 1
	);
}

void VotingPlanner::waypointCallback(const geometry_msgs::PoseStampedConstPtr &msg)
{
    if (!get_cloud)
        return;

    // init begin and end point
    ROS_WARN("Triggered!");
    std::vector<int> pointIdxNKNSearch(1);
	std::vector<float> pointNKNSquaredDistance(1);
    pcl::PointXYZ end_pt(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    pcl::PointXYZ searchPoint(start_pt[0], start_pt[1], start_pt[2]);
	kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
    size_t start_idx = pointIdxNKNSearch[0];
	kdtree.nearestKSearch(end_pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
    size_t end_idx = pointIdxNKNSearch[0];

    ROS_WARN("start=%f, %f, %f", kdcloud->points[start_idx].x, kdcloud->points[start_idx].y,kdcloud->points[start_idx].z);
    ROS_WARN("end=%f, %f, %f", kdcloud->points[end_idx].x, kdcloud->points[end_idx].y,kdcloud->points[end_idx].z);

    // dijkstra and draw
    dijkstra(start_idx, end_idx, KNN);
    drawPath(KNN);
    dijkstra(start_idx, end_idx, SHOTEST);
    drawPath(SHOTEST);
    dijkstra(start_idx, end_idx, TENSORVOTING);
    drawPath(TENSORVOTING);
}

void VotingPlanner::dijkstra(size_t start, size_t end, int type)
{
    ros::Time t1 = ros::Time::now();
	Eigen::VectorXi collected = Eigen::VectorXi::Zero(numPoints);
	Eigen::VectorXi parent = -1 * Eigen::VectorXi::Ones(numPoints);
	Eigen::VectorXf dist = Eigen::VectorXf::Ones(numPoints) * 1e+4;
    Eigen::VectorXf fdist = Eigen::VectorXf::Ones(numPoints) * 1.0e+4;

    dist(start) = 0.0;
    fdist(start) = 0.0;
    
    while (collected(end) == 0)
    {
        Eigen::VectorXf::Index V;
        fdist.minCoeff(&V);
        collected(V) = 1;
        fdist(V) = 1.0e+4;
        std::vector<int> pointIdxNKNSearch(KNN_K);
        std::vector<float> pointNKNSquaredDistance(KNN_K);
        pcl::PointXYZ searchPoint = kdcloud->points[V];
        Eigen::Vector3f V_pt(searchPoint.x, searchPoint.y, searchPoint.z);
        kdtree.nearestKSearch(searchPoint, KNN_K, pointIdxNKNSearch, pointNKNSquaredDistance);
        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
        {
            int W = pointIdxNKNSearch[i];
            if (collected(W) == 0)
            {
                // minimal node number
                float Evw = 1.0;
                // Euclidean distance
                if (type == SHOTEST)
                    Evw = (V_pt - Eigen::Vector3f(kdcloud->points[W].x, kdcloud->points[W].y, kdcloud->points[W].z)).norm();
                // Liu Ming Riemannian metric
                if (type == TENSORVOTING)
                {
                    Eigen::Vector3f diffVec = V_pt - Eigen::Vector3f(kdcloud->points[W].x, kdcloud->points[W].y, kdcloud->points[W].z);
                    float diffNorm = diffVec.norm();
                    Evw = diffVec.norm()*exp(1.0-fabs(diffVec.dot(optdirection.col(V))/diffNorm));
                }

                float new_dist = dist(V) + Evw;
                if (new_dist < dist(W))
                {
                    dist(W) = fdist(W) = new_dist;
                    parent(W) = V;
                }
            }
        }
    }

    ROS_WARN("Find path! cost = %f ms", (ros::Time::now() - t1).toSec() * 1000);

    path.clear();
    int p = end;
    do
    {
        path.push_back(Eigen::Vector3d(kdcloud->points[p].x, kdcloud->points[p].y, kdcloud->points[p].z));
        p = parent(p);
    } while (p != -1);
    path.reserve(path.size());
}

void VotingPlanner::drawPath(int type)
{
    int id;
    Eigen::Vector4d color;
    switch (type)
    {
    case KNN:
        id = 10;
        color = Eigen::Vector4d(1.0, 0.0, 0.0, 0.8);
        break;
    case SHOTEST:
        id = 11;
        color = Eigen::Vector4d(0.0, 1.0, 0.0, 0.8);
        break;
    case TENSORVOTING:
        id = 12;
        color = Eigen::Vector4d(1.0, 1.0, 0.0, 0.8);
        break;
    default:
        break;
    }
    double scale = 0.3;
    visualization_msgs::Marker sphere, line_strip;
    sphere.header.frame_id = line_strip.header.frame_id = "map";
    sphere.header.stamp = line_strip.header.stamp = ros::Time::now();
    sphere.type = visualization_msgs::Marker::SPHERE_LIST;
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;
    sphere.action = line_strip.action = visualization_msgs::Marker::ADD;
    sphere.id = id;
    line_strip.id = id + 1000;

    sphere.pose.orientation.w = line_strip.pose.orientation.w = 1.0;
    sphere.color.r = line_strip.color.r = color(0);
    sphere.color.g = line_strip.color.g = color(1);
    sphere.color.b = line_strip.color.b = color(2);
    sphere.color.a = line_strip.color.a = color(3);
    sphere.scale.x = scale;
    sphere.scale.y = scale;
    sphere.scale.z = scale;
    line_strip.scale.x = scale / 2;
    geometry_msgs::Point pt;
    for (size_t i = 0; i < path.size(); i++)
    {
        pt.x = path[i](0);
        pt.y = path[i](1);
        pt.z = path[i](2);
        sphere.points.push_back(pt);
        line_strip.points.push_back(pt);
    }
    pathPub.publish(sphere);
    pathPub.publish(line_strip);
}

void VotingPlanner::gotCloud(const sensor_msgs::PointCloud2ConstPtr cloudMsgIn)
{
    if (!get_cloud)
    {
        cloud = DP(PointMatcher_ros::rosMsgToPointMatcherCloud<float>(*cloudMsgIn));
        pcl::fromROSMsg(*cloudMsgIn, *kdcloud);
		kdtree.setInputCloud(kdcloud);

        // do sparse tensor voting
        process(cloud, sigma);

        ROS_WARN("output Cloud descriptor pointcloud size: %d", 
                    (unsigned int)(cloud.features.cols()));
        if(savevtk)
        {
            PointMatcherIO<float>::saveVTK(cloud, "/home/xulong/tensor_vote/SavedVTK.vtk");
        }
        get_cloud = true;
    }

    if (cloudPub.getNumSubscribers())
		cloudPub.publish(PointMatcher_ros::pointMatcherCloudToRosMsg<float>(cloud, mapFrame, ros::Time::now()));
}

void VotingPlanner::process(DP & cloud, float sigma)
{
    numPoints = cloud.features.size()/4;
    ROS_WARN("Input size: %d", numPoints);
    ROS_WARN("Sparse ball voting (GPU)...");

    // 1. allocate field
    Eigen::Matrix<Eigen::Matrix3f, Eigen::Dynamic, 1> sparseField;
    size_t sizeField = numPoints*3*sizeof(float3);
    float3* h_fieldarray = (float3 *)malloc(sizeField);

    // 2. allocate points
    size_t sizePoints = numPoints*sizeof(float3);
    float3 *h_points = (float3 *)malloc(sizePoints);
    for(unsigned int i = 0; i<numPoints; i++)
    {
        h_points[i].x = cloud.features(0,i); 
        h_points[i].y = cloud.features(1,i); 
        h_points[i].z = cloud.features(2,i); 
    }
    // 3. set log
    size_t sizeLog = numPoints*sizeof(int2);
    int2 * h_log = (int2 *)malloc(sizeLog);
    bzero( h_log, sizeLog);

    // 4. call CUDA
    ROS_WARN("Send to GPU...");
    CudaVoting::sparseBallVoting(h_fieldarray, h_points, sigma, numPoints, h_log);

    // 5. post-processing
    sparseField.resize(numPoints, 1);
    for(unsigned int i = 0; i<numPoints; i++)
    {
        Eigen::Matrix3f M;
        M << h_fieldarray[i*3 + 0].x, h_fieldarray[i*3 + 0].y, h_fieldarray[i*3 + 0].z, 
             h_fieldarray[i*3 + 1].x, h_fieldarray[i*3 + 1].y, h_fieldarray[i*3 + 1].z, 
             h_fieldarray[i*3 + 2].x, h_fieldarray[i*3 + 2].y, h_fieldarray[i*3 + 2].z;
        sparseField(i) = M;
    }

    // 6. Split on GPU:
    ROS_WARN("sparse tensor split...");
    size_t sizeSaliency = numPoints*sizeof(float);
    float * h_stick = (float*)malloc(sizeSaliency);
    float * h_plate = (float*)malloc(sizeSaliency);
    float * h_ball = (float*)malloc(sizeSaliency);
    // sparse fields
    float3 * h_sparse_stick_field = (float3 *)malloc(numPoints*sizeof(float3));
    float3 * h_sparse_plate_field = (float3 *)malloc(numPoints*sizeof(float3));

    CudaVoting::tensorSplitWithField(h_fieldarray, h_stick, h_plate, h_ball, 
                h_sparse_stick_field, h_sparse_plate_field, numPoints);

    // 7. save sparse tensor
    stick=PM::Matrix::Zero(1, numPoints);
    plate=PM::Matrix::Zero(1, numPoints);
    ball =PM::Matrix::Zero(1, numPoints);
    normals=PM::Matrix::Zero(3, numPoints);
    optdirection=PM::Matrix::Zero(3, numPoints);
    for(unsigned int i=0; i<numPoints; i++)
    {
        stick(i) = h_stick[i];
        plate(i) = h_plate[i];
        ball(i) =  h_ball[i];
        normals.col(i) << h_sparse_stick_field[i].x,h_sparse_stick_field[i].y,h_sparse_stick_field[i].z;
        optdirection.col(i) << h_sparse_plate_field[i].x,h_sparse_plate_field[i].y,h_sparse_plate_field[i].z;
    }
    cloud.addDescriptor("stick", stick);
    cloud.addDescriptor("plate", plate);
    cloud.addDescriptor("ball", ball);
    cloud.addDescriptor("normals", normals);
    cloud.addDescriptor("optdirection", optdirection);

    // 8. clean up
    free(h_fieldarray);
    free(h_points);
    free(h_log);
    free(h_stick);
    free(h_plate);
    free(h_ball);
    free(h_sparse_stick_field);
    free(h_sparse_plate_field);
}


// Main function supporting the SparseVotingCloudGPU class
int main(int argc, char **argv)
{
	ros::init(argc, argv, "test_plan");
	ros::NodeHandle n;
	VotingPlanner votingplan(n);
	ros::spin();
	
	return 0;
}
