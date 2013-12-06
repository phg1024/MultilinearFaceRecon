// RandomTree.cpp : Defines the entry point for the console application.
//



#include "stdafx.h"
#include "randtree.h"

#include "DetectionAAMCombination.h"

#include "AAM_Detection_Combination.h"

char * wch2chr(LPCTSTR lpString)
{
	// Calculate unicode string length.
	UINT len = wcslen(lpString)*2;
	char *buf = (char *)malloc(len);
	UINT i = wcstombs(buf,lpString,len);
	return buf;
}

//void showImage(string dir1,string dir2)
//{
//	string fileName=dir1+"\\imgList.txt";
//	char name[500];
//	char cname[500];
//
//	float x,y;
//	int num;
//	int width,height;
//	int ptsNum;
//	ifstream in(fileName.c_str());
//	in>>num;
//	
//	in.getline(name,500,'\n');
//
//	for (int i=0;i<num;i++)
//	{
//		Mat curImg=imread(name);
//		sprintf(cname, "%d.txt", dir2.c_str(),10000+i);
//		string ptsName=cname;
//		ptsName=ptsName.substr(1,ptsName.length()-1);
//		ptsName=dir2+"\\"+ptsName;
//
//		ifstream in_pts(ptsName.c_str(),ios::in);
//		{
//			in_pts>>width>>height;
//			in_pts>>ptsNum>>ptsNum;
//
//			for (int j=0;j<ptsNum;j++)
//			{
//				in_pts>>x>>y;
//			}
//		}
//	}
//}
int _tmain(int argc, _TCHAR* argv[])
{

	/*Mat tmp=Mat::eye(Size(4,4),CV_64FC1);
	float ii,jj;
	while(1)
	{
		cin>>ii>>jj;
		cout<<tmp.at<double>(ii,jj)<<endl;
	}*/
	
	//12,17,45,13,40,46,20,35,35,25
	//float array[]={12,17,45,13,40,46,20,35,35,25};
	//Mat data=Mat::zeros(1,10,CV_64FC1);
	//for (int i=0;i<10;i++)
	//{
	//	data.at<double>(0,i)=array[i];
	//}


	//GeoHashing *test=new GeoHashing(data,0.25);
	//test->buildHashTabel(test->basisNum,test->basisTabel,data);
	//return 1;
//	cout<<1<<endl;
//	return 1;

	//char name[500];
	//sprintf(name, "AAM+detection/%s_%d.jpg","AAMAFTERLARGEDB2",11361);
	//cout<<name<<endl;

	//AAM, detection, prior
	//AAM_Detection_Combination *engine=new AAM_Detection_Combination(1,.05,0.0001,0.0001);
	//AAM_Detection_Combination *engine=new AAM_Detection_Combination(1,0.05,0.001,0);
	
	//AAM_Detection_Combination *engine=new AAM_Detection_Combination(1,0,0,0);

	string datapath = "C:\\Users\\PhG\\Desktop\\Data\\Fuhao\\model\\";
	string searchPicDir;
	string savePrefix;
	string AAMSearchPrefix;
	string colorRT_model;
	string depthRT_model;
	string AAMModelPath;
	string alignedShapeDir;
	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\Muscle\\kinectColor\\imglist.txt";
	//savePrefix="MuscleEXP";
	//AAMSearchPrefix="MuscleEXP_colorDepth";
	//colorRT_model="D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_color_thres.txt";
	//depthRT_model="D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_depthOnlyThres.txt";

	//searchPicDir="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\Zain\\Zain_imglist.txt";
	//savePrefix="ZainEXP";

	searchPicDir=datapath + "imgList.txt";
	savePrefix="DougTalking_AAM_Sin";

	//searchPicDir="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\Zain\\Zain_imglist.txt";
	//savePrefix="_AAM_Sin_bothCmp_MAX";
	//AAMSearchPrefix=savePrefix;


	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\AndyIllumination_KSeq\\imglist.txt";
	//savePrefix="AndyIllumination_AAM_Sin";
	
	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\DougTalkingComplete_KSeq\\imglist.txt";
	//savePrefix="AAM_Sin";
	//AAMSearchPrefix="AAM_Sin";
	//searchPicDir="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\Garrett\\Garrett_imglist.txt";
	//savePrefix="Garrett_AAM_Sin";
	//AAMSearchPrefix=savePrefix;
	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Alex_Expression_KSeq\\imglist.txt";
	//savePrefix="Alex_Expression_AAM_Sin";
	
	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Kaitlin_Expression_KSeq\\imglist.txt";
	//savePrefix="Kaitlin_Expression_AAM_Sin";
	
	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Joe_Expression_KSeq\\imglist.txt";
	//savePrefix="Joe_Expression_AAM_Sin";
	
	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Joe_talking_KSeq\\imglist.txt";
	//savePrefix="Joe_Talking_AAM_Sin";
	
	//searchPicDir="D:\\Fuhao\\Siggraph\\2013 data\\SDKcomparison\\Muscle 0\\kinect Studio Format\\seq_0\\imglist.txt";
	//searchPicDir="D:\\Fuhao\\Siggraph\\2013 data\\SDKcomparison\\Muscle 0\\kinect Studio Format\\seq_1\\imglist.txt";
	//savePrefix="MuscleSeq0";
	//AAMSearchPrefix="MuscleSeq0";
	/*colorRT_model="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\NoKaitlin\\trainedTree_15_12_56_22_1.txt";
	depthRT_model="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\NoKaitlin\\trainedTree_15_12_56_22_0.txt";
	AAMModelPath="D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\trainedResault_91_90.txt";
	alignedShapeDir="D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_91_90.txt";*/	

	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Fed occlusion 2_KSeq\\imglist.txt";
	//savePrefix="AAM_Sin";
	//AAMSearchPrefix="AAM_Sin";

	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Rhema_Outdoor_KSeq\\imglist.txt";
	//savePrefix="AAM_Sin";
	//AAMSearchPrefix="AAM_Sin";
	
	//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoKaitlin\\trainedTree_15_12_56_22_1.txt";
	//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoKaitlin\\trainedTree_15_12_56_22_0.txt";
	//AAMModelPath="D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\trainedResault_91_90.txt";
	//alignedShapeDir="D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_91_90.txt";

	
	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\FuhaoDepthTest\\imglist.txt";
	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\DougTalkingComplete_KSeq\\imglist.txt";
	//	searchPicDir="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\imglist.txt";
	//savePrefix="Doug_Adapt_OldModel";

	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\AndyIllumination_KSeq\\imglist.txt";
	//savePrefix="Andy_Adapt_OldModel";

	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Alex_Expression_KSeq\\imglist.txt";
	//savePrefix="Alex_Adapt_OldModel";

	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Joe_Expression_KSeq\\imglist.txt";
	//savePrefix="Joe_Adapt_OldModel";

	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Rhema_Outdoor_KSeq\\imglist.txt";
	//savePrefix="Rhema_Adapt";

	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\ICCV comparison\\Fed occlusion 2_KSeq\\imglist.txt";
	//savePrefix="Fed_Adapt";

	//searchPicDir="D:\\Fuhao\\face dataset new\\Biwi\\kinect_head_pose_db\\08\\imglist.txt";
	//savePrefix="08_Adapt";

	//searchPicDir="D:\\Fuhao\\face dataset new\\Biwi\\kinect_head_pose_db\\11\\imglist.txt";
	//savePrefix="11_Adapt";

	//searchPicDir="D:\\Fuhao\\face dataset new\\Biwi\\kinect_head_pose_db\\17\\imglist.txt";
	//savePrefix="17_Adapt";

	//searchPicDir="D:\\Fuhao\\face dataset new\\Biwi\\kinect_head_pose_db\\21\\imglist.txt";
	//savePrefix="21_Adapt";

	/*searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\Tiana_KSeq\\imglist.txt";
	savePrefix="Tiana_Adapt";*/

	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\Tiana_RotTest\\imglist.txt";
	//savePrefix="Tiana_RotTest";
	
	//searchPicDir="D:\\Fuhao\\face dataset\\kinect sequences\\Fuhao_newTest_1024\\imglist.txt";
	//savePrefix="Fuhao_Rot_Adapt";
	//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoKaitlin\\trainedTree_15_12_56_22_1.txt";
	//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoKaitlin\\trainedTree_15_12_56_22_0.txt";

	colorRT_model= datapath + "trainedTree_15_12_56_22_1.txt";
	depthRT_model= datapath + "trainedTree_15_12_64_22_0.txt";


	/*colorRT_model="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\NoGarrett\\trainedTree_15_12_56_22_1.txt";
	depthRT_model="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\NoGarrett\\trainedTree_15_12_56_22_0.txt";*/
	//AAMModelPath="D:\\Fuhao\\face dataset\\train AAM finalEnlarged\\NoPeizhao\\trainedResault_90_90.txt";
	//alignedShapeDir="D:\\Fuhao\\face dataset\\train AAM finalEnlarged\\NoPeizhao\\allignedshape_90_90.txt";

	//AAMModelPath="D:\\Fuhao\\face dataset\\train AAM finalEnlarged\\NoDougTalkingComplete\\trainedResault_90_89.txt";
	//alignedShapeDir="D:\\Fuhao\\face dataset\\train AAM finalEnlarged\\NoDougTalkingComplete\\allignedshape_90_89.txt";



	////original setting
	//colorRT_model="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\NoZain\\trainedTree_15_12_56_22_1.txt";
	//depthRT_model="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\NoZain\\trainedTree_15_12_56_22_0.txt";
	
	//AAMModelPath="D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\trainedResault_91_90.txt";
	//alignedShapeDir="D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_91_90.txt";

	//full complete
	AAMModelPath= datapath + "trainedResult_90_90.txt";
	alignedShapeDir= datapath + "allignedshape_90_90.txt";
	if (argc>3)
	{
		searchPicDir=(argv[1]);
		savePrefix=(argv[2]);
		AAMSearchPrefix=(argv[3]);
		colorRT_model=(argv[4]);
		depthRT_model=(argv[5]);

		if (argc>5)
		{
			AAMModelPath=(argv[6]);
			alignedShapeDir=(argv[7]);
		}
		
		//nameDir=nameDir1;
		//strcpy(nameDir,argv[5])
		searchPicDir+="imglist.txt";
	}


	cout<<searchPicDir<<endl<<savePrefix<<endl<<AAMSearchPrefix<<endl<<colorRT_model<<endl<<depthRT_model<<endl;

	//joe:1,0.5,0.01, down to 0.3, 15 iterations
	//Alex:5,0.5,0.01, not very stable
	//Andy illumination: (1,0.05,0.001, down to 0.005, 30 iterations
	//Doug talking: (1,0.05,0.001, down to 0.005, 30 iterations
	//talking occlusion:0.01 0.5 0.1, down to 0.3, 10 iteration
	//AAM_Detection_Combination *engine=new AAM_Detection_Combination(1,0,0,0,colorRT_model,depthRT_model,AAMModelPath,alignedShapeDir);
	AAM_Detection_Combination *engine=new AAM_Detection_Combination(1,0.005,0.001,0,colorRT_model,depthRT_model,AAMModelPath,alignedShapeDir,true);
	//AAM_Detection_Combination *engine=new AAM_Detection_Combination(1,0.5,0,0,colorRT_model,depthRT_model);
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\Billy expression\\imglist.txt");
	//engine->prepareModel();
	//engine->searchPics("D:\\Fuhao\\face dataset\\images Fuhao test\\imglist.txt");

	//strcpy(engine->AAM_exp->prefix,"TEST_CANDIDATES_PRIOR");
	//engine->searchPics("D:\\Fuhao\\face dataset\\cases test\\imglist.txt");
	
	/*strcpy(engine->AAM_exp->prefix,"TEST_softKNN_LOCALPRIOR_Detection");
	engine->searchPics("D:\\Fuhao\\face dataset\\cases test_old\\imglist.txt");
	return 1;*/
	//strcpy(engine->prefix,"BillyEXP");
	//////strcpy(engine->AAM_exp->prefix,"BillyEXP_colorDepth");
	//strcpy(engine->AAM_exp->prefix,"BillyEXP_separated");
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\Billy expression\\imglist.txt");
	//return 1;
/**/
	//strcpy(engine->prefix,"GarrettEXP");
	////strcpy(engine->AAM_exp->prefix,"GarrettEXP_colorDepth");
	//strcpy(engine->AAM_exp->prefix,"GarrettEXP_separated");
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\Garrett expression\\imglist.txt");

	//strcpy(engine->prefix,"JasonEXP");
	//strcpy(engine->AAM_exp->prefix,"JasonEXP_AAMAFTERLARGEDB2");
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\Jason expression 1\\imglist.txt");
	
	//strcpy(engine->prefix,"DarrenEXP");
	////strcpy(engine->AAM_exp->prefix,"DarrenEXP_colorDepth");
	//strcpy(engine->AAM_exp->prefix,"DarrenEXP_separated");
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\Darren\\2\\kinectColor\\imglist.txt");/**/
	//////	 
	//////
	//////	
	///////**/
	//strcpy(engine->prefix,"DavidEXP");
	//////strcpy(engine->AAM_exp->prefix,"DavidEXP_colorDepth");
	//strcpy(engine->AAM_exp->prefix,"DavidEXP_separated");
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\david\\1\\kinectColor\\imglist.txt");

	//strcpy(engine->prefix,"KaintlinEXP");
	//strcpy(engine->AAM_exp->prefix,"KaintlinEXP_AAMAFTERLARGEDB2");
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\Kaitlin Expression\\imglist.txt");
	//engine->searchPics("D:\\Fuhao\\face dataset\\test_synData\\imglist.txt");

	//strcpy(engine->prefix,"MuscleEXP");
	////strcpy(engine->AAM_exp->prefix,"MuscleEXP_colorDepth");
	//strcpy(engine->AAM_exp->prefix,"MuscleEXP_separated");
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\Muscle\\kinectColor\\imglist.txt");/**/


	strcpy(engine->prefix,savePrefix.c_str());
	//strcpy(engine->AAM_exp->prefix,"MuscleEXP_colorDepth");
	strcpy(engine->AAM_exp->prefix,savePrefix.c_str());
	engine->searchPics(searchPicDir);/**/

	//strcpy(engine->prefix,"RockEXP");
	////strcpy(engine->AAM_exp->prefix,"RockEXP_colorDepth");
	//strcpy(engine->AAM_exp->prefix,"RockEXP_separated_windowsizeColor32MeanShift");
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\Rock\\1\\kinectColor\\imglist.txt");
	
	//strcpy(engine->prefix,"RefineTestEXP");
	//strcpy(engine->AAM_exp->prefix,"RefineTest");
	//engine->searchPics("D:\\Fuhao\\face dataset\\images for refinement testing\\imglist.txt");/**/
	return 1;

	
//	strcpy(engine->AAM_exp->prefix,"BillyEXP_AAMONLY");
//	engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\Billy expression\\imglist.txt");

	
//	strcpy(engine->AAM_exp->prefix,"GarrettEXP_AAMONLY");
//	engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\Garrett expression\\imglist.txt");
//
//	//strcpy(engine->AAM_exp->prefix,"JasonEXP");
//	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\Jason expression 1\\imglist.txt");
//
//	strcpy(engine->AAM_exp->prefix,"DarrenEXP_AAMONLY");
//	engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\Darren\\1\\kinectColor\\imglist.txt");/**/
//////	 
//////
//////	
///////**/
//	strcpy(engine->AAM_exp->prefix,"DavidEXP_AAMONLY");
//	engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\david\\1\\kinectColor\\imglist.txt");
//
//	strcpy(engine->AAM_exp->prefix,"MuscleEXP_AAMONLY");
//	engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\Muscle\\kinectColor\\imglist.txt");/**/

	//strcpy(engine->AAM_exp->prefix,"RockEXP_AAMONLY");
	//engine->searchPics("D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\Rock\\1\\kinectColor\\imglist.txt");

	return 1;



	//DetectionWithAAM *engine=new DetectionWithAAM();
	//engine->AAM_exp->showSingleStep=false;
	//engine->isFindModes=false;
	////engine->searchPics("G:\\face database\\kinect data\\test real data_original image_compact sampling\\imglist.txt");
	////engine->searchPics("D:\\Fuhao\\face dataset\\images Fuhao test\\imglist.txt");
	//engine->searchPics("D:\\Fuhao\\face dataset\\images for combination debug\\imglist.txt");
	////engine->searchPics("G:\\face database\\kinect data\\images for debug\\imglist.txt");
	//return 1;




	//test feature points
	//106 118 137 227 233 250 258 281 327 392
	//RandTree rt1(11,3,0,20,64,7);
	//rt1.imageFeaturePts("F:\\imgdata\\video scale and illumination\\test images\\250_video 4.jpg");
	//return 1;

	//vector<int> tmp;
	//tmp.resize(100);
	//for (int i=0;i<10;i++)
	//{
	//	tmp[i]=i;
	//}

	//for (int i=10;i<100;i++)
	//{
	//	tmp.pop_back();
	//	//cout<<tmp.at(i)<<" ";
	//}

	//for (int i=0;i<10;i++)
	//{
	//	cout<<tmp[i]<<" ";
	//}

	//cout<<tmp.size()<<endl;


	//RandTree rt(15,3,0,21,48,25);
	RandTree rt(15,3,0,17,48,25);
//	RandTree rt(12,3,0,12,80,13);
	/*RandTree rt(15,3,0,15,32,7);
	rt.getSample("F:\\imgdata\\Video 2 Train\\ptsList.txt");
	rt.getSample("F:\\imgdata\\video 4\\train\\ptsList.txt");
	rt.getSample("F:\\imgdata\\oriental Illumination\\train large\\ptsList.txt");*/
	//rt.getSample("F:\\imgdata\\Rock sec20\\web_train\\ptsList.txt");
	//rt.getSample("F:\\imgdata\\Rock sec20\\web_train\\ptsList.txt");
	//rt.getSample("F:\\imgdata\\oriental Illumination\\train large\\ptsList.txt");
	//rt.getSample("F:\\imgdata\\oriental Illumination\\train dynamic evaluation\\ptsList.txt");
	//rt.getSample("F:\\imgdata\\video 4\\train\\ptsList.txt");
	//rt.getSample("f:\\imgdata\\rock sec20\\train_evaluation\\ptslist.txt");
//	rt.trainStyle=0;
//	////rt.getSample("F:\\imgdata\\Rock sec20\\train_evaluation_gradient\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\Rock sec25\\web_train\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\oriental Illumination\\train gradient\\ptsList.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\obama_train\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\obama_train\\transformed\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\mixture train\\transformed\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\Bush\\train\\transformed\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\jim kerry\\train\\transformed\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\Bush\\train 2\\transformed\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\mixture train 2\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\super mixture train\\transformed\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\obama train 2\\transformed\\ptslist.txt");
//	//rt.getSample("F:\\imgdata\\celibrities\\Bush\\train miniminal\\transformed\\ptslist.txt");
//	//rt.getSample("G:\\face database\\BioID\\selected\\train\\ptsList.txt");
//	//rt.getSample("G:\\face database\\lfpw\\train\\transformed\\ptsList.txt");
//	//rt.getSample("G:\\face database\\lfpw\\train\\selected\\transformed\\ptsList.txt");
//	//rt.getSample("G:\\face database\\100 faces depth\\train\\ptsList.txt");
//	//rt.getSample("G:\\face database\\facegen database depth\\train\\ptsList.txt");
//	//rt.getNameList("G:\\face database\\train\\ptsList.txt");
//	//rt.getNameList("G:\\face database\\kinect data\\images me 2\\train_depth\\ptsList.txt");
//	//rt.getNameList("G:\\face database\\train mixture\\ptsList.txt");
//	//rt.getNameList("G:\\face database\\kinect data\\train real data\\ptsList.txt");
//	//rt.getNameList("G:\\face database\\kinect data\\train real data_original image\\ptsList.txt");
//	//rt.getNameList("G:\\face database\\train new\\ptsList.txt");
//	//rt.getNameList("G:\\face database\\kinect data\\train real data_original image_compact sampling\\ptsList.txt");
//	rt.getNameList("G:\\face database\\train mixture new\\ptsList.txt");
//	rt.sampleNumEveryTime=-2;
////	rt.getSample("G:\\face database\\train\\ptsList.txt",100);
////	rt.distroyAllData();
////	rt.getSample("G:\\face database\\train\\ptsList.txt",100);
//	cout<<"training\n";
//	rt.train();
////int interestInd[]={2,3,10,11,22,23,0,1,8,9,24,27,12,13,14,15,16,17,20};
//	cout<<"saving\n";
//	rt.save();
//	return 1;

	//cout<<"loading\n";
	rt.usingCUDA=true;
	//rt.load("F:\\imgdata\\Video 2 Train\\trainedTree_32_threshold.txt");
	//return 0;
	//rt.load("F:\\imgdata\\video 4\\train\\trainedTree.txt");
	//rt.load("F:\\imgdata\\oriental Illumination\\train\\trainedTree.txt");
	////rt.save("F:\\imgdata\\Video 2 Train\\load and saved tree.txt");
	//rt.load("F:\\imgdata\\oriental Illumination\\train large\\trainedTree_48.txt");
	//rt.load("F:\\imgdata\\Rock sec20\\web_train\\trainedTree_92.txt");
	//rt.load("F:\\imgdata\\Rock sec25\\web_train\\trainedTree_48.txt");
	//rt.load("F:\\imgdata\\Rock sec20\\web_train\\trainedTree_15_6_60.txt");
	//rt.load("F:\\imgdata\\oriental Illumination\\train large\\trainedTree_12_15_80.txt");

	///////////////////////
//	//rt.load("F:\\imgdata\\Rock sec20\\train_evaluation\\trainedTree_12_12_60_2_mouth corner.txt");
	//rt.load("F:\\imgdata\\Rock sec20\\train_evaluation_gradient\\trainedTree_12_12_60_13.txt");
	//rt.load("F:\\imgdata\\Rock sec20\\train_evaluation_gradient\\trainedTree_12_12_60_2.txt");
	//rt.load("F:\\imgdata\\Rock sec25\\web_train\\trainedTree_12_12_60_13.txt");
	//rt.load("F:\\imgdata\\Rock sec25\\web_train\\trainedTree_13_12_60_13.txt");
	//rt.load("F:\\imgdata\\Rock sec20\\train_evaluation\\trainedTree_12_12_60.txt");
//	//rt.load("F:\\imgdata\\oriental Illumination\\train dynamic evaluation\\trainedTree_14_12_80.txt");
	//rt.load("F:\\imgdata\\oriental Illumination\\train gradient\\trainedTree_12_12_80_13.txt");
	rt.trainStyle=2;
	//rt.load("F:\\imgdata\\celibrities\\obama_train\\trainedTree_13_12_60_13_style 2.txt");
	//rt.load("F:\\imgdata\\celibrities\\obama_train\\transformed\\trainedTree_13_12_60_13_multi.txt");
	//rt.load("F:\\imgdata\\celibrities\\obama_train\\transformed\\trainedTree_13_12_60_13_multi_more rotations.txt");
	//rt.load("F:\\imgdata\\celibrities\\obama_train\\transformed\\trainedTree_13_12_60_13_multiple orientation.txt");
	//rt.load("F:\\imgdata\\celibrities\\obama_train\\transformed\\traininedTree_13_12_45_13.txt");
	//rt.load("F:\\imgdata\\celibrities\\Bush\\train\\transformed\\trainedTree_13_12_45_13.txt");
	//rt.load("F:\\imgdata\\celibrities\\jim kerry\\train\\transformed\\trainedTree_13_12_45_13.txt");
	//rt.load("F:\\imgdata\\celibrities\\Bush\\train 2\\transformed\\trainedTree_13_13_45_13_threshold.txt");
	//rt.load("F:\\imgdata\\celibrities\\super mixture train\\transformed\\trainedTree_13_13_45_13_binary.txt");
	//rt.load("F:\\imgdata\\celibrities\\super mixture train\\transformed\\trainedTree_13_13_45_13_threshold.txt");
	//rt.load("F:\\imgdata\\celibrities\\obama train 2\\transformed\\trainedTree_13_13_45_13_threshold.txt");
	//rt.load("F:\\imgdata\\celibrities\\obama train 2\\transformed\\trainedTree_13_15_45_13_threshold.txt");
	//rt.load("F:\\imgdata\\celibrities\\Bush\\train miniminal\\transformed\\trainedTree_13_15_45_13_threshold.txt");
	//rt.load("F:\\imgdata\\celibrities\\super mixture train\\transformed\\trainedTree_13_15_45_13_threshold.txt");
	//rt.load("G:\\face database\\BioID\\selected\\train\\trainedTree_13_15_48_13.txt");
	//rt.load("G:\\face database\\lfpw\\train\\transformed\\trainedTree_13_15_52_13.txt");
	//rt.load("G:\\face database\\lfpw\\train\\selected\\transformed\\trainedTree_13_15_45_13.txt");
	//rt.load("F:\\imgdata\\celibrities\\Bush\\train miniminal\\transformed\\trainedTree_13_13_45_13.txt");
	//rt.load("G:\\face database\\lfpw\\train\\selected\\transformed\\trainedTree_15_15_45_26.txt");
	//rt.load("G:\\face database\\lfpw\\train\\selected\\transformed\\trainedTree_15_15_45_29_single scale.txt");
	//rt.load("G:\\face database\\100 faces depth\\train\\trainedTree_25_12_200_25.txt");
	//rt.load("G:\\face database\\100 faces depth\\train\\trainedTree_20_12_200_25.txt");
	//rt.load("G:\\face database\\100 faces depth\\train\\trainedTree_21_12_200_25.txt");
	//rt.load("G:\\face database\\facegen database depth\\train\\trainedTree_21_14_200_25.txt");
	//rt.load("G:\\face database\\train\\trainedTree_20_15_48_25.txt");
	//rt.load("G:\\face database\\kinect data\\images me 2\\train_depth\\trainedTree_20_15_48_25.txt");
	//rt.load("G:\\face database\\train mixture\\trainedTree_20_15_30_25.txt");
	//rt.load("G:\\face database\\kinect data\\train real data\\trainedTree_20_15_48_25.txt");
	//rt.load("G:\\face database\\kinect data\\train real data\\trainedTree_21_15_48_25_depth only more scale.txt");
	//rt.load("G:\\face database\\kinect data\\train real data\\trainedTree_19_15_48_25_depth scaled.txt");
	//rt.load("G:\\face database\\kinect data\\train real data_original image\\trainedTree_15_15_48_25_color only.txt");
	//rt.load("G:\\face database\\train new\\trainedTree_16_15_48_25_depth only.txt");
	//rt.load("G:\\face database\\kinect data\\train real data_original image_compact sampling\\trainedTree_16_15_48_25_depth only.txt");
	rt.load("G:\\face database\\kinect data\\train real data_original image_compact sampling\\trainedTree_17_15_48_25_color_depth.txt");
	
	//rt.load("G:\\face database\\kinect data\\train real data_original image\\trainedTree_16_15_48_25_depth_threshold.txt");
	//rt.load("G:\\face database\\kinect data\\train real data_original image\\trainedTree_20_15_48_25_depth only.txt");
	//rt.load("G:\\face database\\train\\trainedTree_21_15_48_25_treeNum_8.txt");
	//rt.load("G:\\face database\\lfpw\\train\\selected\\transformed\\trainedTree_15_15_45_29_three scale.txt");
	cout<<"predicting\n";
//	
//	//
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\42.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\99.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\118.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\148.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\157.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\282.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\370.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\406.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\461.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\496.jpg");
//	//rt.predict_fulltest("F:\\imgdata\\randomized tree test\\519.jpg");
//
//
//	///////////////////////////test///////////////////////////////////////
//	//rt.predict_imgList("F:\\imgdata\\video 4\\Video 4\\imgList.txt");
//	//rt.predict_imgList("F:\\imgdata\\video 4\\test sample\\imgList.txt");
//	//rt.predict_imgList("F:\\imgdata\\video 7\\train\\imgList.txt");
//	//rt.predict_imgList("F:\\imgdata\\video 7\\test 2\\imgList.txt");
//	//rt.predict_imgList("F:\\imgdata\\rand trees with global constraint test\\imgList.txt");
//	//rt.predict_imgList("F:\\imgdata\\video 4\\Video 4\\imgList.txt");
//	//rt.predict_imgList("F:\\imgdata\\oriental Illumination\\test 90\\imgList.txt");
//	//rt.predict_imgList_fast("F:\\imgdata\\oriental Illumination\\test 32\\imgList.txt");
//	//rt.predict_imgList("F:\\imgdata\\video scale and illumination\\test images\\imgList.txta");
//	//rt.predict_img_transform("F:\\imgdata\\video scale and illumination\\test images not well\\292_video 7.jpg",
//	//	-10,1,false,false);
//	//rt.predict_imgList_fast("F:\\imgdata\\Rock sec25\\test\\imgList.txt");
//	//rt.predict_imgList_fast("F:\\imgdata\\Rock sec20\\test depth and tree num\\imgList.txt");
//	//rt.predict_imgList_fast("F:\\imgdata\\oriental Illumination\\test tree number and depth\\imgList.txt");
//	//rt.predict_imgList_fast("F:\\imgdata\\Rock sec20\\test_evaluation\\imgList.txt");
//	//rt.predict_imgList_fast("F:\\imgdata\\oriental Illumination\\test evaluation\\imgList.txt");
//	//rt.predict_imgList_fast("F:\\imgdata\\Rock sec20\\test single\\imgList.txt");
	//rt.predict_imgList_fast("f:\\imgdata\\rock sec20\\test_evaluation_gradient\\imglist.txt");
	//rt.predict_imgList_fast("f:\\imgdata\\oriental illumination\\test gradient\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\Rock sec25\\rdtree test_2\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\mixture test\\adjusted\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\mixture train\\transformed\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\Bush\\train\\transformed\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\Bush\\train\\transformed\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\Bush\\test\\adjusted\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\jim kerry\\test\\adjusted\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\Bush\\test 2\\adjusted\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\obama test 2\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\obama test_2\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\Bush\\test miniminal\\adjusted\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\super mixture test\\adjusted\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\super mixture test 2\\adjusted\\imglist.txt",true);
	//rt.predict_imgList("F:\\imgdata\\celibrities\\super mixture combination test\\imglist.txt");
	//rt.predict_imgList("G:\\face database\\BioID\\selected\\test_all\\imglist.txt");
	//rt.predict_imgList_fast("F:\\imgdata\\celibrities\\scale test\\imglist.txt");
	//rt.predict_imgList("G:\\face database\\lfpw\\test_train\\imglist.txt");
	//rt.predict_imgList("G:\\face database\\lfpw\\test_sample\\transformed\\imglist.txt");
	//rt.predict_imgList("G:\\face database\\lfpw\\test_19points\\imglist.txt");
	//rt.predict_imgList("G:\\face database\\lfpw\\test_allpoints\\imglist.txt");
	//rt.predict_DepthImgList("G:\\face database\\100 faces depth\\test\\imglist.txt");
	//rt.predict_DepthImgList("G:\\face database\\facegen database depth\\test1\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\test\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\kinect data\\images tiana\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\kinect data\\images me 2\\test_depth\\imglist.txt");
	//rt.predict_DepthImgList("G:\\face database\\test\\imglist.txt");
	//rt.predict_DepthIm gList("G:\\face database\\test\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\test mixture\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\kinect data\\test real data_original image\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\kinect data\\test real data_original image_compact sampling\\imglist.txt");
	rt.predict_DepthList_fast_withLabelCenter("G:\\face database\\kinect data\\test real data_original image_compact sampling\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\kinect data\\test real data_original image_compact sampling\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\test new\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\kinect data\\test real data_original image\\imglist.txt");
	//rt.predict_DepthList_fast("G:\\face database\\kinect data\\images tiana\\train\\imglist.txt");
	return 1;
	/////////////////////

	rt.getSample("f:\\imgdata\\rock sec20\\train_evaluation\\ptslist.txt");
	Mat mat=imread("F:\\imgdata\\Rock sec20\\train_evaluation\\cvtColor_undist_sync_webCam_00000.png");
	CvPoint pos;
	pos.x=rt.shape[8]->pts[0][0];
	pos.y=rt.shape[8]->pts[0][1];

	//for(int i=0;i<40;i++)
	//{
	//	namedWindow("1");
	//	circle(mat,Point(rt.shape[0]->pts[i][0],rt.shape[0]->pts[i][1]),2,Scalar(255));
	//	imshow("1",mat);
	//	waitKey();
	//}

	if (rt.trainStyle==0||rt.trainStyle==1)
	{
		rt.showTree(rt.roots[0],mat,pos);
	}
	else if(rt.trainStyle==2)
	{
		Mat tmp;
		cvtColor(mat,tmp,CV_RGB2GRAY);
		rt.getGradientMap(tmp,tmp);
		/*namedWindow("1");
		imshow("1",tmp);
		waitKey();*/
		rt.showTree(rt.roots[0],mat,tmp,pos);
	}
	

	
	return 0;
}

