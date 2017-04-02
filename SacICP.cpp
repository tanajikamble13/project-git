

/* Point Cloud Registration by Tanaji Kamble using PCL Library*/

#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/registration/ia_ransac.h>
#define POINTTYPE pcl::PointXYZRGB
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//convenient typedefs
//typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointT;//tan change
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

// This is a tutorial so we can afford having global variables 
	//our visualizer
	pcl::visualization::PCLVisualizer *p;
	//its left and right viewports
	int vp_1, vp_2;

//convenient structure to handle our pointclouds
struct PCD
{
  PointCloud::Ptr cloud;
 
  PCD() : cloud (new PointCloud) {};
};

// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
public:
  MyPointRepresentation ()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};


////////////////////////////////////////////////////////////////////////////////
/** Rendering source and target on the first viewport of the visualizer
 *
 */
void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target");
  p->removePointCloud ("vp1_source");

  //PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 0, 255);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> tgt_h (cloud_target);//Tan change
  //PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> src_h (cloud_source);//Tan change
  p->addPointCloud (cloud_target, tgt_h, "vp1_target", vp_1);
  p->addPointCloud (cloud_source, src_h, "vp1_source", vp_1);

  PCL_INFO ("Press q to begin the registration.\n");
  p-> spin();
}


////////////////////////////////////////////////////////////////////////////////
/** Rendering Normalized source and target on the second viewport of the visualizer
 *
 */
void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
  p->removePointCloud ("source");
  p->removePointCloud ("target");


  PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature");
  if (!tgt_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");

  PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature");
  if (!src_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");

  p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2);
  p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);

  p->spinOnce();
}



////////////////////////////////////////////////////////////////////////////////
/** \brief Align a pair of PointCloud datasets and return the result
  * \param cloud_src the source PointCloud
  * \param cloud_tgt the target PointCloud
  * \param output the resultant aligned source PointCloud
  * \param final_transform the resultant transform between source and target
  */
void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
  //
  // Downsample for consistency and speed
  // \note enable this for large datasets
  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid;
  if (downsample)
  {
    grid.setLeafSize (0.05, 0.05, 0.05);
    grid.setInputCloud (cloud_src);
    grid.filter (*src);

    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else
  {
    src = cloud_src;
    tgt = cloud_tgt;
  }


  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
  //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());//Tan change
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);
  
  norm_est.setInputCloud (src);
  norm_est.compute (*points_with_normals_src);
  pcl::copyPointCloud (*src, *points_with_normals_src);

  norm_est.setInputCloud (tgt);
  norm_est.compute (*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

  //
  // Instantiate our custom point representation (defined above) ...
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);

  //
  // Align
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setRANSACOutlierRejectionThreshold( 0.1 ); 
  reg.setTransformationEpsilon (1e-6);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance (1);  
  // Set the point representation
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
  reg.setInputSource (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);


  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations (2);
  for (int i = 0; i < 50; ++i)
  {
    PCL_INFO ("Iteration Nr. %d.\n", i);

    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputSource (points_with_normals_src);
    reg.align (*reg_result);

		//accumulate transformation between each Iteration
    Ti = reg.getFinalTransformation () * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
    prev = reg.getLastIncrementalTransformation ();

    // visualize current state
    showCloudsRight(points_with_normals_tgt, points_with_normals_src);
     std::cout << "has converged:" << reg.hasConverged() << " score: " <<
  reg.getFitnessScore() << std::endl;
  std::cout << reg.getFinalTransformation() << std::endl;
  }

	//
  // Get the transformation from target to source
  targetToSource = Ti.inverse();

  //
  // Transform target back in source frame
 pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);

  p->removePointCloud ("source");
  p->removePointCloud ("target");

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb3 (output);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb4 (cloud_src);
  p->addPointCloud<pcl::PointXYZRGB>(cloud_src, rgb4, "source", vp_2);
  p->addPointCloud<pcl::PointXYZRGB>(output, rgb3, "target", vp_2);

  //PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0);
  //PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
  //p->addPointCloud (output, cloud_tgt_h, "target", vp_2);
  //p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);
  //PCL_INFO ("Press q to continue the registration.\n");

  p->spin ();

  p->removePointCloud ("source"); 
  p->removePointCloud ("target");

  //add the source to the transformed target
  *output += *cloud_src;
  
  final_transform = targetToSource;

 }


/* ---[ */
int main (int argc, char** argv)
{
  // Object for storing the point cloud.
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourceCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
 
	// Read a PCD file from disk.
	if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[1], *sourceCloud) != 0)
	{
		return -1;//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
	}
       if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[2], *targetCloud) != 0)
	{
		return -1;
	}

       boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Source cloud (Left) Target cloud (Right)"));
        viewer->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
        viewer->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
        viewer->removePointCloud ("vp1_target");
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb (sourceCloud);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2 (targetCloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(sourceCloud, rgb, "source", vp_1);
        viewer->addPointCloud<pcl::PointXYZRGB>(targetCloud, rgb2, "target", vp_2);
        viewer-> spin();

  pcl::PointCloud<POINTTYPE>::Ptr cloud1 (new pcl::PointCloud<POINTTYPE>);
  pcl::PointCloud<POINTTYPE>::Ptr cloud2 (new pcl::PointCloud<POINTTYPE>);

  *cloud1= *sourceCloud;
  *cloud2= *targetCloud;
  //pcl::io::loadPCDFile<POINTTYPE> (argv[1], *cloud1);
  //pcl::io::loadPCDFile<POINTTYPE> (argv[2], *cloud2);

  std::vector<int> ind,ind2;
  pcl::removeNaNFromPointCloud(*cloud1,*cloud1,ind);
  pcl::removeNaNFromPointCloud(*cloud2,*cloud2,ind2);
  printf("size:%d %d\n",cloud1->size(),cloud2->size());
  
		pcl::PointCloud<pcl::PointNormal>::Ptr norm1(new pcl::PointCloud<pcl::PointNormal>);
		pcl::PointCloud<pcl::PointNormal>::Ptr norm2(new pcl::PointCloud<pcl::PointNormal>);
		pcl::search::KdTree<POINTTYPE>::Ptr normSM(new pcl::search::KdTree<POINTTYPE>);
		pcl::NormalEstimation<POINTTYPE, pcl::PointNormal> norm_est;
		norm_est.setSearchMethod (normSM);
		norm_est.setRadiusSearch (0.05);
		norm_est.setInputCloud(cloud1);
		norm_est.compute(*norm1);
		norm_est.setInputCloud(cloud2);
		norm_est.compute(*norm2);
		  for(size_t i = 0; i<norm1->points.size(); ++i)
    norm1->points[i].x = cloud1->points[i].x,
    norm1->points[i].y = cloud1->points[i].y,
    norm1->points[i].z = cloud1->points[i].z;
		  for(size_t i = 0; i<norm2->points.size(); ++i)
    norm2->points[i].x = cloud2->points[i].x,
    norm2->points[i].y = cloud2->points[i].y,
    norm2->points[i].z = cloud2->points[i].z;

		pcl::PointCloud<pcl::PointWithScale> siftresult1;
		pcl::PointCloud<pcl::PointWithScale> siftresult2;
		//sift rgb
		//pcl::SIFTKeypoint<POINTTYPE, pcl::PointWithScale> sift_detect;pcl::search::KdTree<POINTTYPE>::Ptr siftkdtree(new pcl::search::KdTree<POINTTYPE> ());sift_detect.setScales (0.01, 3, 4);sift_detect.setMinimumContrast (0.001);

		//sift normal
		//pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift_detect;pcl::search::KdTree<pcl::PointNormal>::Ptr siftkdtree(new pcl::search::KdTree<pcl::PointNormal> ());sift_detect.setScales (0.1, 6, 10);sift_detect.setMinimumContrast (0.5);

		//sift z
 pcl::SIFTKeypoint<POINTTYPE, pcl::PointWithScale> sift_detect;
 pcl::search::KdTree<POINTTYPE>::Ptr siftkdtree(new pcl::search::KdTree<POINTTYPE> ());

 //pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> sift_detect;
 //pcl::search::KdTree<pcl::PointXYZ>::Ptr siftkdtree(new pcl::search::KdTree<pcl::PointXYZ> ());

 sift_detect.setScales (0.005, 6, 4);
 sift_detect.setMinimumContrast (0.005);

		sift_detect.setSearchMethod	(siftkdtree);
		sift_detect.setInputCloud (cloud1);//sift rgb/z
		//sift_detect.setInputCloud (norm1);//sift normal
		sift_detect.compute (siftresult1);
                printf("keypoint1 computed ,size:%d\n",siftresult1.size());
		sift_detect.setInputCloud (cloud2);//sift rgb/z
		//sift_detect.setInputCloud (norm2);//sift normal
		sift_detect.compute (siftresult2);
                printf("keypoint2 computed ,size:%d\n",siftresult2.size());

		pcl::PointCloud<POINTTYPE> keypoint1;
                pcl::copyPointCloud(siftresult1,keypoint1);
		pcl::PointCloud<POINTTYPE> keypoint2;
                pcl::copyPointCloud(siftresult2,keypoint2);
                
                
                
                /*
                boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("Source KeyPoint (Left) Target KeyPoint (Right)"));
        viewer2->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
        viewer2->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
        viewer2->removePointCloud ("vp1_target");
        //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> c1 (keypoint1.makeshared());
       // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> c2 (keypoint1.makeshared());
       viewer2->setBackgroundColor(0,0,0,vp_1);
       viewer2->setBackgroundColor(0.5,0.5,0.5,vp_2);
      // pv.setBackgroundColor(0.5,0.5,0.5,v3);
	viewer2->addPointCloud<pcl::PointXYZRGB>(keypoint1,  "source", vp_1);
        viewer2->addPointCloud<pcl::PointXYZRGB>(keypoint2,  "target", vp_2);
        viewer2-> spin();
        
        
        */
        
  /*      
  pcl::visualization::PCLVisualizer pv("C");
  //pv.addPointCloud(cloud1);
  pv.createViewPort(0.0, 0, 0.5, 1.0, vp_1);
  pv.createViewPort(0.5, 0, 1.0, 1.0, vp_2);
 
  pv.setBackgroundColor(0,0,0,v1);
  pv.setBackgroundColor(0.5,0.5,0.5,v2);
  pv.setBackgroundColor(0.5,0.5,0.5,v3);
  pv.setBackgroundColor(1,1,1,v4);
  
  pv.addPointCloud(keypoint1.makeShared(),"a",vp_1);
  pv.addPointCloud(keypoint2.makeShared(),"b",vp_2);
  pv.spin();
  */   

//pcl::visualization::PCLVisualizer pva("A");pva.setBackgroundColor(0.5,0.5,0.5);pcl::visualization::PCLVisualizer pvb("B");pvb.setBackgroundColor(0.5,0.5,0.5);pva.addPointCloud(keypoint1.makeShared());pvb.addPointCloud(keypoint2.makeShared());while(!pva.wasStopped()&&!pvb.wasStopped())pva.spinOnce(),pvb.spinOnce();
		
#define FEATUREDESCRIPTOR pcl::FPFHSignature33
	  
      pcl::PointCloud<FEATUREDESCRIPTOR>::Ptr descriptor1(new pcl::PointCloud<FEATUREDESCRIPTOR>);
      pcl::PointCloud<FEATUREDESCRIPTOR>::Ptr descriptor2(new pcl::PointCloud<FEATUREDESCRIPTOR>);
      pcl::FPFHEstimation<POINTTYPE,pcl::PointNormal, FEATUREDESCRIPTOR> feature_est;
      feature_est.setSearchMethod (pcl::search::Search<POINTTYPE>::Ptr (new pcl::search::KdTree<POINTTYPE>));
      feature_est.setRadiusSearch (0.2f);//too small will crash
      feature_est.setSearchSurface(cloud1);
      feature_est.setInputNormals(norm1);
      feature_est.setInputCloud (keypoint1.makeShared());
      feature_est.compute (*descriptor1);
      printf("descriptor1 computed\n");
      feature_est.setSearchSurface(cloud2);
      feature_est.setInputNormals(norm2);
      feature_est.setInputCloud (keypoint2.makeShared());
      feature_est.compute (*descriptor2);
      printf("descriptor2 computed\n");
     
		pcl::SampleConsensusInitialAlignment<POINTTYPE, POINTTYPE, FEATUREDESCRIPTOR> sacia;
		sacia.setMinSampleDistance (0.05);
		sacia.setMaxCorrespondenceDistance (0.2);
		sacia.setMaximumIterations (500);
                 /*
		sacia.setInputCloud (cloud1);
		sacia.setInputTarget (cloud2);
		sacia.setSourceFeatures (descriptor1);
		sacia.setTargetFeatures (descriptor2);
                   */
                //It should be 
               
                sacia.setInputCloud (keypoint1.makeShared());
                sacia.setInputTarget (keypoint2.makeShared());
                sacia.setSourceFeatures (descriptor1);
                sacia.setTargetFeatures (descriptor2); 
             
		//pcl::PointCloud<POINTTYPE> Final;
		  pcl::PointCloud<POINTTYPE>::Ptr Final(new pcl::PointCloud<POINTTYPE>());
		sacia.align (*Final);
               // pcl::io::savePCDFileASCII ("final.pcd", Final());

		cout<<"\n has converged:" << sacia.hasConverged() << " Fitness Score: " << sacia.getFitnessScore();
		cout<<"\nTransformation:\n" << sacia.getFinalTransformation();
                /*
		pcl::visualization::PCLVisualizer pv("RANSAC Registration");
		pv.addPointCloud(Final.makeShared());
		pv.spin();

                */




  // Create a PCLVisualizer object
  p = new pcl::visualization::PCLVisualizer (argc, argv, "ICP Registration ");
  p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
  p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);

  PointCloud::Ptr result (new PointCloud), source, target, resultant;
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform;
  
  //pcl::PointCloud<POINTTYPE>::Ptr FinalCloud (new pcl::PointCloud<POINTTYPE>);
 
   //*FinalCloud= Final;
    //transform current pair into the global transform
    pcl::transformPointCloud (*sourceCloud, *result, sacia.getFinalTransformation());
    source = result;
    target = targetCloud;

    // Add visualization data
    showCloudsLeft(source, target);

    PointCloud::Ptr temp (new PointCloud);
   
    pairAlign (source, target, temp, pairTransform, true);

    //transform current pair into the global transform
    pcl::transformPointCloud (*temp, *result, GlobalTransform);

    //update the global transform
    GlobalTransform = GlobalTransform * pairTransform;
    pcl::transformPointCloud (*sourceCloud, *resultant, GlobalTransform);
 /*
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer3 (new pcl::visualization::PCLVisualizer("Final"));
        //viewer3->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
       // viewer3->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
        //viewer3->removePointCloud ("vp1_target");
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb3 (resultant);
        //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2 (targetCloud);
        viewer3->addPointCloud<pcl::PointXYZRGB>(targetCloud, rgb2, "target");
        viewer3->addPointCloud<pcl::PointXYZRGB>(resultant, rgb3, "final"); 
*/
}
/* ]--- */




