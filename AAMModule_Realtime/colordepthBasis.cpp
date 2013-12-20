//------------------------------------------------------------------------------
// <copyright file="ColorBasics.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#include "stdafx.h"
#include <strsafe.h>
#include "coloranddepthRec.h"
#include "resource.h"
#include <fstream>
#include <vector>

#include <stdio.h>
#include <io.h>
#include <fcntl.h>
using namespace std;

//LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
//LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
//{
//	PAINTSTRUCT ps;
//
//	application.HandleMessages(hWnd, message, wParam, lParam);
//
//	/*switch( message )
//	{
//	case WM_PAINT:
//	BeginPaint(hWnd, &ps);
//	EndPaint(hWnd, &ps);
//	break;
//
//	case WM_DESTROY:
//	PostQuitMessage(0);
//	break;      
//
//	default:
//	return DefWindowProc(hWnd, message, wParam, lParam);
//	}
//	*/
//	return 0;
//}

/// <summary>
/// Entry point for the application
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="hPrevInstance">always 0</param>
/// <param name="lpCmdLine">command line arguments</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
/// <returns>status</returns>
int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
{
	AllocConsole();

	HANDLE handle_out = GetStdHandle(STD_OUTPUT_HANDLE);
	int hCrt = _open_osfhandle((long) handle_out, _O_TEXT);
	FILE* hf_out = _fdopen(hCrt, "w");
	setvbuf(hf_out, NULL, _IONBF, 1);
	*stdout = *hf_out;

	HANDLE handle_in = GetStdHandle(STD_INPUT_HANDLE);
	hCrt = _open_osfhandle((long) handle_in, _O_TEXT);
	FILE* hf_in = _fdopen(hCrt, "r");
	setvbuf(hf_in, NULL, _IONBF, 128);
	*stdin = *hf_in;

	CColorDepthBasics application;
	application.Run(hInstance, nCmdShow);
}

/// <summary>
/// Constructor
/// </summary>
CColorDepthBasics::CColorDepthBasics() :
	m_pD2DFactory(NULL),
	m_pDrawColor(NULL),
	m_hNextColorFrameEvent(INVALID_HANDLE_VALUE),
	m_hNextDepthFrameEvent(INVALID_HANDLE_VALUE),
	m_pColorStreamHandle(INVALID_HANDLE_VALUE),
	m_pDepthStreamHandle(INVALID_HANDLE_VALUE),
	m_bSaveScreenshot(false),
	m_pNuiSensor(NULL)
{
	// create heap storage for depth pixel data in RGBX format
	m_depthRGBX = new BYTE[cDepthWidth*cDepthHeight*cBytesPerPixel];
	m_depthD16 = new USHORT[cDepthWidth*cDepthHeight];
	m_colorCoordinates = new LONG[cDepthWidth*cDepthHeight*2];
	m_colorRGBX = new BYTE[cColorWidth*cColorHeight*cBytesPerPixel];
	m_colorRGBXAligned=new BYTE[cColorWidth*cColorHeight*cBytesPerPixel];
	//depthData=new float[cColorWidth*cColorHeight];

	outputVideoOrigin=false;
	outputVideoAligned=false;
	outputDepth=false;
	outputDepthWarped=false;

	showDetecton=false;
	usingOriginDepth=false;

	totalFrameNo=1100;
	//depthData1=new float[totalFrameNo*cColorWidth*cColorHeight];

	if (outputDepth)
	{
		depthData1=new vector<float>[totalFrameNo];
		for (int i=0;i<totalFrameNo;i++)
		{
			depthData1[i].resize(cColorWidth*cColorHeight);
		}
	}

	if (outputVideoOrigin)
	{
		videoOrign.resize(totalFrameNo);
		for (int i=0;i<totalFrameNo;i++)
		{
			videoOrign[i]=new BYTE [cColorWidth*cColorHeight*cBytesPerPixel];
		}
	}

	if (outputVideoAligned)
	{
		videoAligned.resize(totalFrameNo);
		for (int i=0;i<totalFrameNo;i++)
		{
			videoAligned[i]=new BYTE [cColorWidth*cColorHeight*cBytesPerPixel];
		}
	}

	if (outputDepthWarped)
	{
		depthImagesWarped=new Mat[totalFrameNo];
		for (int i=0;i<totalFrameNo;i++)
		{
			depthImagesWarped[i].create(cColorHeight,cColorWidth,CV_32FC1);
		}
	}


	m_colorToDepthDivisor=cColorWidth/cDepthWidth;

	showOriginalImage=false;

	capture=false;
	frameId=0;
	cframeId=0;
	startInd=100;
	endInd=100+totalFrameNo;

	initial=true;
	curStatus=0;
	startNum=0;
	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);

	//rt=new RandTree(15,3,0,16,48,25);
	//rt->usingCUDA=false;
	/*rt->trainStyle=2;
	rt->load("G:\\face database\\kinect data\\train real data\\trainedTree_15_15_48_25_depth color.txt");*/

	//rt->trainStyle=0;
	//rt->load("G:\\face database\\kinect data\\train real data\\trainedTree_20_15_48_25 depth only.txt");
	//rt->load("G:\\face database\\kinect data\\train real data\\trainedTree_21_15_48_25_depth only more scale.txt");

	////using original depth
	//rt->trainStyle=0;
	//rt->load("G:\\face database\\kinect data\\train real data\\trainedTree_19_15_48_25_depth scaled.txt");

	//using warped depth
	//rt->trainStyle=0;
	//rt->load("G:\\face database\\kinect data\\train real data_original image\\trainedTree_20_15_48_25_depth only.txt");

	//rt->trainStyle=1;
	//rt->load("G:\\face database\\kinect data\\train real data_original image\\trainedTree_15_15_48_25_color only.txt");

	//rt->trainStyle=2;
	//rt->load("G:\\face database\\kinect data\\train real data_original image\\trainedTree_16_15_48_25_color and depth.txt");

	//compact data now
	//rt->trainStyle=0;
	//rt->load("G:\\face database\\kinect data\\train real data_original image_compact sampling\\trainedTree_16_15_48_25_depth only.txt");

	//rt->trainStyle=1;
	//rt->load("G:\\face database\\kinect data\\train real data_original image_compact sampling\\trainedTree_15_15_48_25_color.txt");

	//rt->trainStyle=2;
	//rt->load("G:\\face database\\kinect data\\train real data_original image_compact sampling\\trainedTree_17_15_48_25_color_depth.txt");

	//synthesized data
	//rt->trainStyle=0;
	//rt->load("G:\\face database\\train new\\trainedTree_16_15_48_25_depth only.txt");

	//mixture
	//rt->trainStyle=0;
	//rt->load("G:\\face database\\train\\trainedTree_20_15_48_25_mixture_depth.txt");

	//rt->trainStyle=0;
	//rt->load("G:\\face database\\train\\trainedTree_21_15_48_25_treeNum_8.txt");

	if(1)
	{
		string searchPicDir;
		string savePrefix;
		string AAMSearchPrefix;
		string colorRT_model;
		string depthRT_model;
		string AAMModelPath;
		string alignedShapeDir;

		const string datapath = "C:\\Users\\Peihong\\Desktop\\Data\\Fuhao\\model\\";
		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoGarrett\\trainedTree_15_12_56_22_1.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoGarrett\\trainedTree_15_12_56_22_0.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoGarrett\\trainedTree_15_12_64_22_0.txt";

		//64,64 
		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\trainedTree_15_12_56_22_1.txt";
		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\trainedTree_15_12_64_22_1.txt";
		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\trainedTree_15_12_80_22_1.txt";

		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\trainedTree_15_12_56_22_0.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\trainedTree_15_12_64_22_0.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\trainedTree_15_12_80_22_0.txt";

		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\label_2\\right_15_12_56_22_1.txt";
		//depthRT_model="D:\\Fuhao\face dataset\\train_RGBD_finalEnlarged\\label_2\\right_15_12_56_22_0.txt";

		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\label_3\\left_15_12_56_22_1.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\label_3\\left_15_12_56_22_0.txt";

		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoKaitlin\\trainedTree_15_12_56_22_1.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoKaitlin\\trainedTree_15_12_56_22_0.txt";

		colorRT_model=datapath + "trainedTree_15_12_56_22_1.txt";
		depthRT_model=datapath + "trainedTree_15_12_64_22_0.txt";

		AAMModelPath=datapath + "trainedResult_90_90.txt";
		alignedShapeDir=datapath + "allignedshape_90_90.txt";
		

		//AAMModelPath="D:\\Fuhao\\face dataset\\train AAM finalEnlarged\\NoDougTalkingComplete\\trainedResault_90_89.txt";
		//alignedShapeDir="D:\\Fuhao\\face dataset\\train AAM finalEnlarged\\NoDougTalkingComplete\\allignedshape_90_89.txt";
		//color 80+depth 56 or 64
		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\rightMid\\trainedTree_15_12_56_22_1.txt";
		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\rightMid\\trainedTree_15_12_64_22_1.txt";
		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\rightMid\\trainedTree_15_12_80_22_1.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\rightMid\\trainedTree_15_12_56_22_0.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\rightMid\\trainedTree_15_12_64_22_0.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\rightMid\\trainedTree_15_12_80_22_0.txt";//mouth is the best when training data is more concentrated
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\rightMid\\partialSingle\\trainedTree_15_12_64_22_0.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\rightMid\\partialAll\\trainedTree_15_12_64_22_0_treeNum_8.txt";

		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\leftMid\\trainedTree_15_12_56_22_1.txt";
		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\leftMid\\trainedTree_15_12_64_22_1.txt";
		//colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\leftMid\\trainedTree_15_12_80_22_1.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\leftMid\\trainedTree_15_12_56_22_0.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\leftMid\\trainedTree_15_12_64_22_0.txt";
		//depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged_Reflectory\\leftMid\\trainedTree_15_12_80_22_0.txt";

		/*colorRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoPeizhao\\trainedTree_15_12_56_22_1.txt";
		depthRT_model="D:\\Fuhao\\face dataset\\train_RGBD_finalEnlarged\\NoPeizhao\\trainedTree_15_12_56_22_0.txt";
		AAMModelPath="D:\\Fuhao\\face dataset\\train AAM finalEnlarged\\NoPeizhao\\trainedResault_90_90.txt";
		alignedShapeDir="D:\\Fuhao\\face dataset\\train AAM finalEnlarged\\NoPeizhao\\allignedshape_90_90.txt";*/

		//AAMModelPath="D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\trainedResault_91_90.txt";
		//alignedShapeDir="D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_91_90.txt";

		/*colorRT_model="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\NoGarrett\\trainedTree_15_12_56_22_1.txt";
		depthRT_model="D:\\Fuhao\\face dataset\\train_RBGD_enlarged_030813\\NoGarrett\\trainedTree_15_12_56_22_0.txt";*/


		engine=new AAM_Detection_Combination(0.8,.05,0.005,0,colorRT_model,depthRT_model,AAMModelPath,alignedShapeDir,true);
		//engine=new AAM_Detection_Combination(0.9,0.05,0.003,0,colorRT_model,depthRT_model,AAMModelPath,alignedShapeDir,true);
		//engine=new AAM_Detection_Combination(1,0,0,0,colorRT_model,depthRT_model,AAMModelPath,alignedShapeDir);
		//m_pDrawColor->setPtsInfo(currentShape,78)

	}

	colorImage.create(cColorHeight,cColorWidth,CV_8UC4);
	depthImage.create(cDepthHeight,cDepthWidth,CV_32FC1);
	//	depthImageWarped.create(cDepthHeight,cDepthWidth,CV_32FC1);

	colorImageFlip=colorImage.clone();

	standardDepth=750;

	haveTable=false;
	isRecording=false;
}

/// <summary>
/// Destructor

/// </summary>
CColorDepthBasics::~CColorDepthBasics()
{
	if (m_pNuiSensor)
	{
		m_pNuiSensor->NuiShutdown();
	}

	if (m_hNextColorFrameEvent != INVALID_HANDLE_VALUE)
	{
		CloseHandle(m_hNextColorFrameEvent);
	}

	// clean up Direct2D renderer
	delete m_pDrawColor;
	m_pDrawColor = NULL;

	// clean up Direct2D
	SafeRelease(m_pD2DFactory);

	SafeRelease(m_pNuiSensor);
}

//writetoMat
void CColorDepthBasics::writeMat(float *depthInfo,WCHAR *lpszFile,int id)
{

	//int   nLen   =   wcslen(lpszFile)+1;   
	//char   *path=new char[nLen]; 
	//WideCharToMultiByte(CP_ACP,   0,   lpszFile,   nLen,   path,   2*nLen,   NULL,   NULL); 
	//// open MAT file
	//MATFile *matf =matOpen(path, "w7.3");
	//if (matf == NULL)
	//{
	//	printf("Error opening file %s\n", path);
	//	return;
	//}

	//int totalNum=cColorWidth*cColorHeight;
	//int cind;
	//if (id==-1)
	//{
	//	mxArray *depthPoints_mx = mxCreateCellMatrix(1, totalFrameNo);
	//	vector<mxArray*> tmpMX_pnt(totalFrameNo);
	//	for (int t=0; t<totalFrameNo; t++)
	//	{
	//		tmpMX_pnt[t] = mxCreateDoubleMatrix(cColorHeight, cColorWidth, mxREAL);
	//		//		tmpMX_pnt[t] = mxCreateDoubleMatrix(0,0,mxREAL);
	//		//		mxSetM (tmpMX_pnt[t], 3);
	//		//		mxSetN (tmpMX_pnt[t], depthPoints[t].size());
	//		//		mxSetPr(tmpMX_pnt[t], &depthPoints[t][0][0]);
	//		double *tmp = mxGetPr(tmpMX_pnt[t]);
	//		for(int ii = 0; ii < cColorWidth; ii++)
	//		{
	//			for(int jj = 0; jj < cColorHeight; jj++)
	//			{
	//				cind=ii + jj * cColorWidth;
	//				*tmp++ = depthData1[t][cind];
	//			}
	//		}
	//		mxSetCell(depthPoints_mx, t, tmpMX_pnt[t]);
	//		
	//	}
	//	//	depth_points_.reserve(0);
	//	//	depth_points_.clear();
	//	matPutVariable(matf, "depthPoints", depthPoints_mx);

	//	mxDestroyArray(depthPoints_mx);
	//}
	//else
	//{
	//	mxArray *depthArray = mxCreateDoubleMatrix(cColorHeight, cColorWidth, mxREAL);
	//	//memcpy(mxGetPr(depthArray), depthData1+id*totalNum, totalNum * sizeof(float));
	//	double *tmp = mxGetPr(depthArray);
	//	for(int ii = 0; ii < cColorWidth; ii++)
	//	{
	//		for(int jj = 0; jj < cColorHeight; jj++)
	//		{
	//			cind=ii + jj * cColorWidth;
	//			*tmp++ = depthData1[id][cind];
	//		}
	//	}

	//	//vector<mxArray*> tmpMX_pnt(totalFrameNo);
	//	//for (int t=0; t<totalFrameNo; t++)
	//	//{
	//	//	tmpMX_pnt[t] = 
	//	//	//		tmpMX_pnt[t] = mxCreateDoubleMatrix(0,0,mxREAL);
	//	//	//		mxSetM (tmpMX_pnt[t], 3);
	//	//	//		mxSetN (tmpMX_pnt[t], depthPoints[t].size());
	//	//	//		mxSetPr(tmpMX_pnt[t], &depthPoints[t][0][0]);
	//	//	double *tmp = mxGetPr(tmpMX_pnt[t]);
	//	//	for(int ii = 0; ii < cColorWidth; ii++)
	//	//	{
	//	//		for(int jj = 0; jj < cColorHeight; jj++)
	//	//		{
	//	//			cind=ii + jj * cColorWidth;
	//	//			*tmp++ = depthData1[t*totalNum+cind];
	//	//		}
	//	//	}
	//	//	mxSetCell(depthPoints_mx, t, tmpMX_pnt[t]);

	//	//}
	//	//	depth_points_.reserve(0);
	//	//	depth_points_.clear();
	//	matPutVariable(matf, "depthPoints", depthArray);
	//	mxDestroyArray(depthArray);
	//}


	////// indexField
	////mxArray *index = mxCreateCellMatrix(1, writeFrameCount);
	////vector<mxArray*> tmpMX_field(writeFrameCount);
	////for (int t=0; t<writeFrameCount; t++)
	////{
	////	//		tmpMX_field[t] = mxCreateDoubleMatrix(0,0,mxREAL);
	////	//		mxSetM (tmpMX_field[t], indexField[t].NumCols());
	////	//		mxSetN (tmpMX_field[t], indexField[t].NumRows());
	////	tmpMX_field[t] = mxCreateDoubleMatrix(index_field_[t].NumCols(), index_field_[t].NumRows(), mxREAL);
	////	double *tmp = mxGetPr(tmpMX_field[t]);
	////	index_field_[t].CopyToArray(tmp);
	////	//		mxSetPr(tmpMX_field[t], indexField[t].GetArray());
	////	mxSetCell(index, t, tmpMX_field[t]);
	////	index_field_[t].SetEmpty();
	////}
	//////	index_field_.reserve(0);
	////index_field_.clear();
	////matPutVariable(matf, "indexField", index);

	////// depthMap
	////mxArray *dp = mxCreateCellMatrix(1, writeFrameCount);
	////vector<mxArray*> tmpMX_depthMap(writeFrameCount);
	////for (int t=0; t<writeFrameCount; t++)
	////{
	////	//tmpMX_depthMap[t] = mxCreateDoubleMatrix(0,0,mxREAL);
	////	//mxSetM (tmpMX_depthMap[t], depthMap[t].NumCols());
	////	//mxSetN (tmpMX_depthMap[t], depthMap[t].NumRows());
	////	tmpMX_depthMap[t] = mxCreateDoubleMatrix(depth_map_[t].NumCols(), depth_map_[t].NumRows(), mxREAL);
	////	double *tmp = mxGetPr(tmpMX_depthMap[t]);
	////	depth_map_[t].CopyToArray(tmp);
	////	//		mxSetPr(tmpMX_depthMap[t], depthMap[t].GetArray());
	////	mxSetCell(dp, t, tmpMX_depthMap[t]);
	////	depth_map_[t].Resize(0, 0);
	////	depth_map_[t].SetEmpty();
	////}
	//////	depth_map_.reserve(0);
	////depth_map_.clear();
	////matPutVariable(matf, "depthMap", dp);

	////// colorMap
	////mxArray *cp = mxCreateCellMatrix(1, writeFrameCount);
	////vector<mxArray*> tmpMX_colorMap(writeFrameCount);
	////for (int t=0; t<writeFrameCount; t++)
	////{
	////	//tmpMX_depthMap[t] = mxCreateDoubleMatrix(0,0,mxREAL);
	////	//mxSetM (tmpMX_depthMap[t], depthMap[t].NumCols());
	////	//mxSetN (tmpMX_depthMap[t], depthMap[t].NumRows());
	////	tmpMX_colorMap[t] = mxCreateDoubleMatrix(color_map_[t].NumCols(), color_map_[t].NumRows(), mxREAL);
	////	double *tmp = mxGetPr(tmpMX_colorMap[t]);
	////	color_map_[t].CopyToArray(tmp);
	////	//		mxSetPr(tmpMX_depthMap[t], depthMap[t].GetArray());
	////	mxSetCell(cp, t, tmpMX_colorMap[t]);
	////	color_map_[t].Resize(0, 0);
	////	color_map_[t].SetEmpty();
	////}
	//////	depth_map_.reserve(0);
	////color_map_.clear();
	////matPutVariable(matf, "colorMap", cp);



	//// view
	////mxArray* view_mx = vp_->ToArray();
	////matPutVariable(matf, "view", view_mx);

	//// close
	//if (matClose(matf) != 0)
	//	printf("Error closing file %s\n",path);

	//delete []path;;

	////mxDestroyArray(depthPoints_mx);

	////mxDestroyArray(index);

	////mxDestroyArray(dp);

	////mxDestroyArray(cp);



	////mxDestroyArray(view_mx);

	//return;
}

LRESULT CColorDepthBasics::HandleMessages(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	//m_camera.HandleMessages(hWnd, uMsg, wParam, lParam);

	switch(uMsg)
	{
		// handle minimize
		//case WM_SIZE:          
		//	if (SIZE_MINIMIZED == wParam)
		//	{
		//		m_bPaused = true;
		//	}
		//	break;

		//	// handle restore from minimized
		//case WM_ACTIVATEAPP:
		//	if (wParam == TRUE)
		//	{
		//		m_bPaused = false;
		//	}
		//	break;

	case WM_KEYDOWN:
		{
			int nKey = static_cast<int>(wParam);

			if (nKey == 'c')
			{
				showOriginalImage=!showOriginalImage;
			}
			else if (nKey=='0')
			{
				cout<<"state changed\n";
				engine->state=0;
			}
			else if (nKey=='1')
			{
				engine->state=1;
			}
			else if (nKey=='2')
			{
				engine->state=2;
			}
			break;
		}
	}

	return 0;
}
/// <summary>
/// Creates the main window and begins processing
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
int CColorDepthBasics::Run(HINSTANCE hInstance, int nCmdShow)
{
	MSG       msg = {0};
	WNDCLASS  wc;

	// Dialog custom window class
	ZeroMemory(&wc, sizeof(wc));
	wc.style         = CS_HREDRAW | CS_VREDRAW;
	wc.cbWndExtra    = DLGWINDOWEXTRA;
	wc.hInstance     = hInstance;
	wc.hCursor       = LoadCursorW(NULL, IDC_ARROW);
	wc.hIcon         = LoadIconW(hInstance, MAKEINTRESOURCE(IDI_APP));
	wc.lpfnWndProc   = DefDlgProcW;
	wc.lpszClassName = L"ColorBasicsAppDlgWndClass";

	if (!RegisterClassW(&wc))
	{
		return 0;
	}

	// Create main application window
	HWND hWndApp = CreateDialogParamW(
		hInstance,
		MAKEINTRESOURCE(IDD_APP),
		NULL,
		(DLGPROC)CColorDepthBasics::MessageRouter, 
		reinterpret_cast<LPARAM>(this));


	// Create main application window
	//HWND hWndApp1 = CreateDialogParamW(
	//	hInstance,
	//	MAKEINTRESOURCE(IDD_APP),
	//	NULL,
	//	(DLGPROC)CColorDepthBasics::MessageRouter, 
	//	reinterpret_cast<LPARAM>(this));

	// Show window
	ShowWindow(hWndApp, nCmdShow);

	// Show window
	//	ShowWindow(hWndApp1, nCmdShow);

	const int eventCount = 1;
	HANDLE hEvents[eventCount];

	// Main message loop
	while (WM_QUIT != msg.message)
	{
		//hEvents[0] = m_hNextColorFrameEvent;
		//hEvents[1] = m_hNextDepthFrameEvent;
		// Check to see if we have either a message (by passing in QS_ALLINPUT)
		// Or a Kinect event (hEvents)
		// Update() will check for Kinect events individually, in case more than one are signalled
		//DWORD dwEvent = MsgWaitForMultipleObjects(eventCount, hEvents, FALSE, INFINITE, QS_ALLINPUT);

		// Check if this is an event we're waiting on and not a timeout or message
		//	if (WAIT_OBJECT_0 == dwEvent)
		//{
		//	
		//}

		if (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
		{
			// If a dialog message will be taken care of by the dialog proc
			if ((hWndApp != NULL) && IsDialogMessageW(hWndApp, &msg))
			{
				continue;
			}

			TranslateMessage(&msg);
			DispatchMessageW(&msg);
		}
		else if(GetAsyncKeyState(VK_UP) & 0x8000)
		{
			//showOriginalImage=true;
			showDetecton=true;
		}
		else if (GetAsyncKeyState(VK_LEFT) & 0x8000)
		{
			//cout<<"state changed\n";
			engine->state=0;
			cout<<"show single frame\n";
		}
		else if(GetAsyncKeyState(VK_DOWN) & 0x8000)
		{
			engine->state=1;
			cout<<"show temporal frame\n";
		}
		else if(GetAsyncKeyState(VK_RIGHT) & 0x8000)
		{
			engine->state=2;
			//frameId=0;
		}
		else if(GetAsyncKeyState(VK_NUMPAD8) & 0x8000)
		{
			engine->isAAMOnly=true;
			if (engine->isAAMOnly)
			{
				cout<<"AAM only\n";
			}
			else
			{
				cout<<"full term\n";
			}
			//frameId=0;
		}
		else if(GetAsyncKeyState(VK_NUMPAD9) & 0x8000)
		{
			engine->isAAMOnly=false;
			if (engine->isAAMOnly)
			{
				cout<<"AAM only\n";
			}
			else
			{
				cout<<"full term\n";
			}
			//frameId=0;
		}
		else if (GetAsyncKeyState(VK_NUMPAD1) & 0x8000)
		{
			engine->showNN=!engine->showNN;
			if (engine->showNN)
			{
				cout<<"showing nearest neighbor\n";
			}
			else
			{
				cout<<"not showing nearest neighbor\n";
			}
		}
		else if (GetAsyncKeyState(VK_NUMPAD2) & 0x8000)
		{
			engine->TemporalTracking=true;
			cout<<"detection using temporal prior\n";
		}
		else if (GetAsyncKeyState(VK_NUMPAD3) & 0x8000)
		{
			engine->TemporalTracking=false;
			cout<<"detection using single frame\n";
		}
		else if (GetAsyncKeyState(VK_NUMPAD4) & 0x8000)
		{
			engine->showProbMap=!engine->showProbMap;
			//cout<<"detection using single frame\n";
		}
		else if (GetAsyncKeyState(VK_NUMPAD5) & 0x8000)
		{
			isRecording=true;
			//cout<<"detection using single frame\n";
		}
		else if (GetAsyncKeyState(VK_NUMPAD6) & 0x8000)
		{
			isRecording=false;
			cout<<"outputing pts\n";
			ofstream out("ptsRec.txt",ios::out);
			for (int i=0;i<ptsList.size();i++)
			{
				out<<ptsList[i]<<" ";
			}
			out.close();
			exit(0);
			//cout<<"detection using single frame\n";
		}

		else if(GetAsyncKeyState(VK_RIGHT) & 0x8000)
		{
			if(capture&&frameId>0)
			{
				//string preFix="Fuhao";
				TCHAR* preFix = TEXT("Garrett 1");

				TCHAR *dirName=TEXT("D:\\Fuhao\\face dataset\\kinect sequences\\garrett 1_new");
				//TCHAR* preFix = TEXT("Peizhao");
				//TCHAR* preFix = TEXT("Yenlin");
				WCHAR statusMessage[cStatusMessageMaxLen];

				// Retrieve the path to My Photos
				WCHAR screenshotPath[MAX_PATH];
				StringCchPrintfW(screenshotPath,  _countof(screenshotPath), L"%s\\%s_%d.bmp", dirName,preFix,10000+frameId);
				//GetScreenshotFileName(screenshotPath, _countof(screenshotPath));

				// Write out the bitmap to disk
				//hr = SaveBitmapToFile(m_colorRGBX, cColorWidth, cColorHeight, 32, screenshotPath);
				//hr = SaveBitmapToFile(m_colorRGBXAligned, cColorWidth, cColorHeight, 32, screenshotPath);
				if (outputDepth)
				{
					for (int i=0;i<frameId;i++)
					{
						int totalNum=cColorWidth*cColorHeight;
						StringCchPrintfW(screenshotPath,  _countof(screenshotPath), L"%s\\%s_%d_depth.txt",dirName,preFix, 10000+i+realStartInd);
						//writeMat(depthData1,screenshotPath,i);
						ofstream out(screenshotPath,ios::out);
						for (LONG y = 0; y < cColorHeight; ++y)
						{
							//LONG* pDest = (LONG*)((BYTE*)msT.pData + msT.RowPitch * y);
							for (LONG x = 0; x < cColorWidth; ++x)
							{
								// calculate index into depth array
								int depthIndex = x + y * cColorWidth;
								//if (depthData1[i][depthIndex]!=0)
								{
									out<<depthData1[i][depthIndex]<<" ";
								}


							}
							out<<endl;
						}
						out.close();
					}
				}

				if (outputVideoOrigin)
				{
					for (int i=0;i<frameId;i++)
					{
						// Retrieve the path to My Photos
						WCHAR screenshotPath[MAX_PATH];
						StringCchPrintfW(screenshotPath,  _countof(screenshotPath), L"%s\\%s_%d.bmp",dirName,preFix, 10000+i+realStartInd);
						//GetScreenshotFileName(screenshotPath, _countof(screenshotPath));

						// Write out the bitmap to disk
						HRESULT hr = SaveBitmapToFile(videoOrign[i], cColorWidth, cColorHeight, 32, screenshotPath);
						//hr = SaveBitmapToFile(m_colorRGBXAligned, cColorWidth, cColorHeight, 32, screenshotPath);
					}

				}

				if (outputVideoAligned)
				{
					for (int i=0;i<frameId;i++)
					{
						// Retrieve the path to My Photos
						WCHAR screenshotPath[MAX_PATH];
						StringCchPrintfW(screenshotPath,  _countof(screenshotPath), L"%s\\%s_%d_aligned.bmp",dirName,preFix, 10000+i+realStartInd);
						//GetScreenshotFileName(screenshotPath, _countof(screenshotPath));

						// Write out the bitmap to disk
						HRESULT hr = SaveBitmapToFile(videoAligned[i], cColorWidth, cColorHeight, 32, screenshotPath);
						//hr = SaveBitmapToFile(m_colorRGBXAligned, cColorWidth, cColorHeight, 32, screenshotPath);
					}

				}

				if (outputDepthWarped)
				{
					for (int i=0;i<frameId;i++)
					{
						WCHAR screenshotPath[MAX_PATH];
						StringCchPrintfW(screenshotPath,  _countof(screenshotPath), L"%s\\%s_%d.xml",dirName,preFix, 10000+i+realStartInd);

						WCHAR pngPath[MAX_PATH];
						StringCchPrintfW(pngPath,  _countof(pngPath), L"%s\\%s_%d.png",dirName,preFix, 10000+i+realStartInd);
						int tmp;
						int a,b,c;
						Mat curPNG=Mat::zeros(480,640,CV_8UC3);
						for (int j=0;j<curPNG.rows;j++)
						{
							for (int k=0;k<curPNG.cols;k++)
							{
								tmp=depthImagesWarped[i].at<float>(j,k)*standardDepth;
								a=tmp/(256*256);
								b=(tmp-a*256*256)/256;
								c=tmp-a*256*256-b*256;
								curPNG.at<Vec3b>(j,k)[0]=a;
								curPNG.at<Vec3b>(j,k)[1]=b;
								curPNG.at<Vec3b>(j,k)[2]=c;
							}
						}
						char strDes1[MAX_PATH];
						wcstombs(strDes1, pngPath, MAX_PATH);
						imwrite(strDes1,curPNG);

						char strDes[MAX_PATH];
						wcstombs(strDes, screenshotPath, MAX_PATH);
						CvMat tmpDepth=depthImagesWarped[i];
						cvSave(strDes, &tmpDepth); 
					}

				}

				capture=false;
			}
		}
		else
			Update();

		//if (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
		//{
		//	// If a dialog message will be taken care of by the dialog proc
		//	if ((hWndApp1 != NULL) && IsDialogMessageW(hWndApp1, &msg))
		//	{
		//		continue;
		//	}

		//	TranslateMessage(&msg);
		//	DispatchMessageW(&msg);
		//}
		//else
		//	Update();
	}

	return static_cast<int>(msg.wParam);
}

/// <summary>
/// Main processing function
/// </summary>
void CColorDepthBasics::Update()
{
	//if (cframeId%500==0)
	//{
	//	QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 
	//	cframeId=0;
	//}
	/*if (showDetecton)
	{
	GTB("START");
	}*/

	if (NULL == m_pNuiSensor)
	{
		return;
	}

	//	QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 


	bool needToMapColorToDepth=false;
	if ( WAIT_OBJECT_0 == WaitForSingleObject(m_hNextDepthFrameEvent, 0) )
	{
		// if we have received any valid new depth data we may need to draw
		if ( (HRESULT)(ProcessDepth())>=0 )
		{
			needToMapColorToDepth = true;
		}
	}

	if ( WAIT_OBJECT_0 == WaitForSingleObject(m_hNextColorFrameEvent, 0) )
	{
		// if we have received any valid new color data we may need to draw
		if ( (HRESULT)(ProcessColor())>=0 )
		{
			needToMapColorToDepth = true;
		}
	}


	if (!m_bDepthReceived || !m_bColorReceived)
	{
		needToMapColorToDepth = false;
	}

	if (needToMapColorToDepth)
	{
		ProcessDepthwithMapping();
		if (capture&&cframeId>startInd&&frameId<totalFrameNo)
		{
			frameId++;
		}
		//if (capture)
		{
			cframeId++;
		}
	}


}

/// <summary>
/// Process color data received from Kinect
/// </summary>
/// <returns>S_OK for success, or failure code</returns>
HRESULT CColorDepthBasics::MapColorToDepth()
{
	HRESULT hr;

	// Get of x, y coordinates for color in depth space
	// This will allow us to later compensate for the differences in location, angle, etc between the depth and color cameras
	m_pNuiSensor->NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution(
		NUI_IMAGE_RESOLUTION_640x480,
		NUI_IMAGE_RESOLUTION_640x480,
		cDepthWidth*cDepthHeight,
		m_depthD16,
		cDepthWidth*cDepthHeight*2,
		m_colorCoordinates
		);

	// copy to our d3d 11 color texture
	//D3D11_MAPPED_SUBRESOURCE msT;
	//	hr = m_pImmediateContext->Map(m_pColorTexture2D, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &msT);
	if ( FAILED(hr) ) { return hr; }

	// loop over each row and column of the color
	for (LONG y = 0; y < cColorHeight; ++y)
	{
		//LONG* pDest = (LONG*)((BYTE*)msT.pData + msT.RowPitch * y);
		for (LONG x = 0; x < cColorWidth; ++x)
		{
			// calculate index into depth array
			int depthIndex = x/m_colorToDepthDivisor + y/m_colorToDepthDivisor * cDepthWidth;

			// retrieve the depth to color mapping for the current depth pixel
			LONG colorInDepthX = m_colorCoordinates[depthIndex * 2];
			LONG colorInDepthY = m_colorCoordinates[depthIndex * 2 + 1];

			// make sure the depth pixel maps to a valid point in color space
			if ( colorInDepthX >= 0 && colorInDepthX < cColorWidth && colorInDepthY >= 0 && colorInDepthY < cColorHeight )
			{
				// calculate index into color array
				LONG colorIndex = colorInDepthX + colorInDepthY * cColorWidth;

				// set source for copy to the color pixel
				LONG* pSrc = (LONG *)m_colorRGBX + colorIndex;
				//*pDest = *pSrc;
			}
			else
			{
				//*pDest = 0;
			}

			//pDest++;
		}
	}

	//m_pImmediateContext->Unmap(m_pColorTexture2D, NULL);

	return hr;
}

/// <summary>
/// Handles window messages, passes most to the class instance to handle
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CColorDepthBasics::MessageRouter(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	CColorDepthBasics* pThis = NULL;

	if (WM_INITDIALOG == uMsg)
	{
		pThis = reinterpret_cast<CColorDepthBasics*>(lParam);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
	}
	else
	{
		pThis = reinterpret_cast<CColorDepthBasics*>(::GetWindowLongPtr(hWnd, GWLP_USERDATA));
	}

	if (pThis)
	{
		return pThis->DlgProc(hWnd, uMsg, wParam, lParam);
	}

	return 0;
}

/// <summary>
/// Handle windows messages for the class instance
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CColorDepthBasics::DlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_INITDIALOG:
		{
			// Bind application window handle
			m_hWnd = hWnd;

			// Init Direct2D
			D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &m_pD2DFactory);

			// Create and initialize a new Direct2D image renderer (take a look at ImageRenderer.h)
			// We'll use this to draw the data we receive from the Kinect to the screen
			m_pDrawColor = new ImageRenderer();
			m_pDrawColor->setPtsInfo(currentShape,engine->ptsNum);
			//m_pDrawColor->pts=engine->currentShape;
			//m_pDrawColor->ptsNum=engine->ptsNum;
			HRESULT hr = m_pDrawColor->Initialize(GetDlgItem(m_hWnd, IDC_VIDEOVIEW), m_pD2DFactory, cColorWidth, cColorHeight, cColorWidth * sizeof(long));
			if (FAILED(hr))
			{
				SetStatusMessage(L"Failed to initialize the Direct2D draw device.");
			}

			m_pDrawDepth = new ImageRenderer();
			hr = m_pDrawDepth->Initialize(GetDlgItem(m_hWnd, IDC_VIDEOVIEW), m_pD2DFactory, cDepthWidth, cDepthHeight, cDepthWidth * sizeof(long));
			if (FAILED(hr))
			{
				SetStatusMessage(L"Failed to initialize the Direct2D draw device.");
			}

			// Look for a connected Kinect, and create it if found
			CreateFirstConnected();
		}
		break;

		// If the titlebar X is clicked, destroy app
	case WM_CLOSE:
		DestroyWindow(hWnd);
		break;

	case WM_DESTROY:
		// Quit the main message pump
		PostQuitMessage(0);
		break;

		// Handle button press
	case WM_COMMAND:
		// If it was for the screenshot control and a button clicked event, save a screenshot next frame 
		if (IDC_BUTTON_SCREENSHOT == LOWORD(wParam) && BN_CLICKED == HIWORD(wParam))
		{
			m_bSaveScreenshot = true;
		}
		break;

	case WM_KEYDOWN:
		{
			int nKey = static_cast<int>(wParam);

			if (nKey == 'c')
			{
				showOriginalImage=!showOriginalImage;
			}
			break;
		}

	}

	return FALSE;
}

/// <summary>
/// Create the first connected Kinect found 
/// </summary>
/// <returns>indicates success or failure</returns>
HRESULT CColorDepthBasics::CreateFirstConnected()
{
	INuiSensor * pNuiSensor;
	HRESULT hr;

	int iSensorCount = 0;
	hr = NuiGetSensorCount(&iSensorCount);
	if (FAILED(hr))
	{
		return hr;
	}

	// Look at each Kinect sensor
	for (int i = 0; i < iSensorCount; ++i)
	{
		// Create the sensor so we can check status, if we can't create it, move on to the next
		hr = NuiCreateSensorByIndex(i, &pNuiSensor);
		if (FAILED(hr))
		{
			continue;
		}

		// Get the status of the sensor, and if connected, then we can initialize it
		hr = pNuiSensor->NuiStatus();
		if (S_OK == hr)
		{
			m_pNuiSensor = pNuiSensor;
			break;
		}

		// This sensor wasn't OK, so release it since we're not using it
		pNuiSensor->Release();
	}

	if (NULL != m_pNuiSensor)
	{
		// Initialize the Kinect and specify that we'll be using color
		hr = m_pNuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR|NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX); 
		if (SUCCEEDED(hr))
		{
			// Create an event that will be signaled when color data is available
			m_hNextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

			// Open a color image stream to receive color frames
			hr = m_pNuiSensor->NuiImageStreamOpen(
				NUI_IMAGE_TYPE_COLOR,
				NUI_IMAGE_RESOLUTION_640x480,
				0,
				2,
				m_hNextColorFrameEvent,
				&m_pColorStreamHandle);

			if (FAILED(hr) ) { return hr; }

			// Create an event that will be signaled when depth data is available
			m_hNextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

			// Open a depth image stream to receive depth frames
			hr = m_pNuiSensor->NuiImageStreamOpen(
				NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX,
				NUI_IMAGE_RESOLUTION_640x480,
				0,
				2,
				m_hNextDepthFrameEvent,
				&m_pDepthStreamHandle);
			if (FAILED(hr) ) { return hr; }
		}
	}

	if (NULL == m_pNuiSensor || FAILED(hr))
	{
		SetStatusMessage(L"No ready Kinect found!");
		return E_FAIL;
	}

	if ( m_pNuiSensor )
	{
		hr = m_pNuiSensor->NuiImageStreamSetImageFrameFlags(m_pDepthStreamHandle, NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE);

		/*if ( SUCCEEDED(hr) )
		{
		m_bNearMode = !m_bNearMode;
		}*/
	}

	return hr;
}

/// <summary>
/// Get the name of the file where screenshot will be stored.
/// </summary>
/// <param name="screenshotName">
/// [out] String buffer that will receive screenshot file name.
/// </param>
/// <param name="screenshotNameSize">
/// [in] Number of characters in screenshotName string buffer.
/// </param>
/// <returns>
/// S_OK on success, otherwise failure code.
/// </returns>
HRESULT GetScreenshotFileName(wchar_t *screenshotName, UINT screenshotNameSize)
{
	wchar_t *knownPath = NULL;
	HRESULT hr = SHGetKnownFolderPath(FOLDERID_Pictures, 0, NULL, &knownPath);

	if (SUCCEEDED(hr))
	{
		// Get the time
		wchar_t timeString[MAX_PATH];
		GetTimeFormatEx(NULL, 0, NULL, L"hh'-'mm'-'ss", timeString, _countof(timeString));

		// File name will be KinectSnapshot-HH-MM-SS.wav
		StringCchPrintfW(screenshotName, screenshotNameSize, L"%s\\KinectSnapshot-%s.bmp", knownPath, timeString);
	}

	CoTaskMemFree(knownPath);
	return hr;
}

void CColorDepthBasics::ProcessDepthwithMapping()
{
	//if (capture&&frameId>=totalFrameNo)
	//{
	//	return;
	//}
	if (showDetecton)
	{
		GTB("START");
	}

	HRESULT hr;

	// Get of x, y coordinates for color in depth space
	// This will allow us to later compensate for the differences in location, angle, etc between the depth and color cameras
	if (!haveTable||(haveTable&&curStatus==1&&engine->AAM_exp->isAdaptive)||(haveTable&&startNum%5==4))
		//if (!haveTable||(haveTable&&startNum%5==3))
			//if(!haveTable)
	{
		//GTB("START");

		m_pNuiSensor->NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution(
			NUI_IMAGE_RESOLUTION_640x480,
			NUI_IMAGE_RESOLUTION_640x480,
			cDepthWidth*cDepthHeight,
			m_depthD16,
			cDepthWidth*cDepthHeight*2,
			m_colorCoordinates
			);
		haveTable=true;

		/*	GTE("START");

		gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"used time per iteration: "<<time<<endl;*/
	}



	// copy to our d3d 11 color texture
	//D3D11_MAPPED_SUBRESOURCE msT;
	//	hr = m_pImmediateContext->Map(m_pColorTexture2D, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &msT);
	//	if ( FAILED(hr) ) { return hr; }
	//GTB("alignment");


	BYTE * rgbrun = m_colorRGBXAligned;	
	//for (LONG y = 0; y < cColorHeight; ++y)
	//{
	//	//LONG* pDest = (LONG*)((BYTE*)msT.pData + msT.RowPitch * y);
	//	for (LONG x = 0; x < cColorWidth; ++x)
	//	{
	//		//// calculate index into depth array
	//		int depthIndex = x/m_colorToDepthDivisor + y/m_colorToDepthDivisor * cDepthWidth;
	//		////depthData[depthIndex]=0;
	//		//if (frameId<totalFrameNo)
	//		//{
	//		//	depthData1[frameId][depthIndex]=0;
	//		//}
	//		

	//			LONG* pSrc = (LONG *)m_colorRGBXAligned + depthIndex;

	//			BYTE* ptr = (BYTE*)pSrc;

	//			// Write out blue byte
	//			*(ptr++) = 0;

	//			// Write out green byte
	//			*(ptr++) = 0;

	//			// Write out red byte
	//			*(ptr++) = 0;

	//			// We're outputting BGR, the last byte in the 32 bits is unused so skip it
	//			// If we were outputting BGRA, we would write alpha here.
	//			++ptr;
	//	}
	//}

	//if (showDetecton)
	//{
	//	colorImage*=0;
	//}

	Mat depthImageWarped=Mat::zeros(cDepthHeight,cDepthWidth,CV_32FC1);
	//colorImage*=0;
	// loop over each row and column of the color
	//	QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 

	//Mat colorImage(640,480,CV_8UC3,(BYTE*)m_colorRGBX);

	//for(int lll=0;lll<100;lll++)
	{
		//GTB("START");
		//#pragma omp parallel for
		//uchar* color_ptr = colorImage.ptr<uchar>();

		for (LONG y = 0; y < cColorHeight; ++y)
		{
			//LONG* pDest = (LONG*)((BYTE*)msT.pData + msT.RowPitch * y);
			for (LONG x = 0; x < cColorWidth; ++x)
			{

				int depthIndex = x + y * cDepthWidth;
				//obtain color image
				//if (showDetecton&&!usingOriginDepth)
				//{
				//	LONG* pSrc = (LONG *)m_colorRGBX + depthIndex;

				//	BYTE* ptr = (BYTE*)pSrc;
				//	//colorImage.at<Vec3b>(y,x).val[0]=static_cast<uchar>(*(ptr+0));
				//	//colorImage.at<Vec3b>(y,x).val[1]=static_cast<uchar>(*(ptr+1));
				//	//colorImage.at<Vec3b>(y,x).val[2]=static_cast<uchar>(*(ptr+2));
				//	memcpy(color_ptr, ptr, 3 * sizeof(uchar));
				//	color_ptr += 3;
				//}
				// calculate index into depth array


				// set source for copy to the color pixel
				USHORT* dSrc = (USHORT *)m_depthD16 + depthIndex;
				USHORT depth = NuiDepthPixelToDepth(* dSrc);

				if (depth==0)
				{
					continue;
				}

				// retrieve the depth to color mapping for the current depth pixel
				LONG colorInDepthX = m_colorCoordinates[depthIndex * 2];
				LONG colorInDepthY = m_colorCoordinates[depthIndex * 2 + 1];

				//		// make sure the depth pixel maps to a valid point in color space
				if ( colorInDepthX >= 0 && colorInDepthX < cColorWidth && colorInDepthY >= 0 && colorInDepthY < cColorHeight )
				{
					depthImageWarped.at<float>(colorInDepthY,colorInDepthX)=depth/standardDepth;//==0?0:1200-depth;				
				}
			}
		}
		//GTE("START");
	}

	//cout<<colorImage.at<Vec3b>(0,0).val[0]<<depthImageWarped.at<float>(0,0);
	//	QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 

	//GTB("START");
	////take 2ms per frame. Seems not necessary now
	medianBlur(depthImageWarped,depthImageWarped,3);

	/*GTE("alignment");

	gCodeTimer.printTimeTree();
	double time = total_fps;
	cout<<"used time per iteration: "<<time<<endl;*/

	//GTE("START");

	if (outputDepthWarped&&frameId<totalFrameNo)
	{
		//depthImagesWarped[frameId]=depthImageWarped.clone();
		//memcpy(videoOrign[frameId], m_colorRGBX,cDepthWidth * cDepthHeight*cBytesPerPixel);	
	}


	if (showDetecton)
	{

		//// loop over each row and column of the color
		//for (LONG y = 0; y < cColorHeight; ++y)
		//{
		//	//LONG* pDest = (LONG*)((BYTE*)msT.pData + msT.RowPitch * y);
		//	for (LONG x = 0; x < cColorWidth; ++x)
		//	{
		//		// calculate index into depth array
		//		LONG colorIndex = x + y * cColorWidth;

		//		LONG* pSrc = (LONG *)m_colorRGBXAligned + colorIndex;

		//		BYTE* ptr = (BYTE*)pSrc;
		//		colorImage.at<Vec3b>(y,x).val[0]=* ptr++;
		//		colorImage.at<Vec3b>(y,x).val[1]=* ptr++;
		//		colorImage.at<Vec3b>(y,x).val[2]=* ptr++;
		//	}
		//}
		/*	if (usingOriginDepth)
		rt->detect(colorImage,depthImage);
		else
		rt->detect(colorImage,depthImageWarped);*/
		/*	namedWindow("1",0);
		imshow("1",colorImage);*/
		//imwrite("test.jpg",colorImage);
		//showDetecton=false;
		Mat colorIMG_Gray;
		cvtColor(colorImage, colorIMG_Gray, CV_BGR2GRAY);
		/*	namedWindow("1",0);
		imshow("1",depthImageWarped);*/
		int cptsNum=engine->AAM_exp->meanShape->ptsNum;
		if (initial)
		{
			//IplImage *img=&((IplImage)colorIMG_Gray);
			//CvSeq* faces = cvHaarDetectObjects( img, engine->face_cascade, engine->faces_storage,
			//	1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
			//	cvSize(30, 30) );

			////printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
			//for( int i = 0; i < (faces ? faces->total : 0); i++ )
			//{
			//	CvRect* r = (CvRect*)cvGetSeqElem( faces, i );

			//	if (r->width>50)
			//	{
			//		startX=r->x;
			//		startY=r->y;
			//		endX=startX+r->width;
			//		endY=startY+r->width+15;
			//	}
			//}
			startX=640-348;
			endX=640-205;
			startY=153;
			endY=289;

			/*startX=640-348;
			endX=640-205;
			startY=153;
			endY=289;*/
			//initial=false;

		}
		else
		{
			startX=startY=100000;
			endX=endY=-1;

			int ccx,ccy;
			if (startNum<2||engine->state==2||1)
			{
				for (int i=0;i<cptsNum;i++)
				{
					ccx=engine->currentShape[i];
					ccy=engine->currentShape[i+cptsNum];
					if (ccx==0||ccy==0)
					{
						continue;
					}
					if (ccx<startX)
					{
						startX=ccx;
					}
					if (ccx>endX)
					{
						endX=ccx;
					}

					if (ccy<startY)
					{
						startY=ccy;
					}
					if (ccy>endY)
					{
						endY=ccy;
					}
				}
			}
			else
			{
				int fullIndNum=engine->fullIndNum;
				for (int i=0;i<fullIndNum;i++)
				{
					ccx=engine->pridictedPts[i];
					ccy=engine->pridictedPts[i+fullIndNum];
					if (ccx<startX)
					{
						startX=ccx;
					}
					if (ccx>endX)
					{
						endX=ccx;
					}

					if (ccy<startY)
					{
						startY=ccy;
					}
					if (ccy>endY)
					{
						endY=ccy;
					}
				}
			}

			startX-=10;
			startY-=10;
			endX+=10;
			endY+=10;
		}

		if (startX<50||startY<50||endX>cColorWidth-50||endY>cColorHeight-50)
		{
			//reset parameters
			initial=true;
			startNum=0;
			engine->hasVelocity=false;
			depthImageWarped.release();
			return;
		}
		/*if (usingOriginDepth)
		engine->track_combine(colorIMG_Gray,depthImage);
		else
		engine->track_combine(colorIMG_Gray,depthImageWarped);*/
		//if (cframeId%5!=0)
		//{

		/*	if (startNum%500==0)
		{
		QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 
		startNum=0;
		}*/

		/*if (showDetecton)
		{
		GTB("START");
		}*/

		bool isSecceed=false;
		bool isDrop=false;
		//if (((curStatus==0&&engine->AAM_exp->isAdaptive))||(!engine->AAM_exp->isAdaptive&&startNum%5!=4))
		//GTB("START");
		//if (((curStatus==0&&engine->AAM_exp->isAdaptive)&&(startNum%5!=4))||(!engine->AAM_exp->isAdaptive&&startNum%5!=4))
		if(1)
			//if (startNum%5!=3)
		{
			//if (!(engine->AAM_exp->isAdaptive&&curStatus==1))
			{
				//GTB("START");
			}

			isSecceed=engine->track_combine(colorIMG_Gray,depthImageWarped,curStatus,startX,endX,startY,endY,startNum>0);
			//isSecceed=engine->track_combine(colorIMG_Gray,depthImageWarped,startX,endX,startY,endY,startNum>0);
			if (initial&&isSecceed)
			{
				initial=false;
			}

			if (!isSecceed)//reset all the parameters to initial status
			{
				initial=true;
				startNum=0;
				engine->hasVelocity=false;
			}

			/*		if (!isSecceed)
			{
			cout<<"failed!\n";
			}*/
			//if (!(engine->AAM_exp->isAdaptive&&curStatus==1))
			{
				/*	GTE("START");

				gCodeTimer.printTimeTree();
				double time = total_fps;
				cout<<"used time per iteration: "<<time<<" ms"<<endl;*/
			}

		}
		//if ((curStatus==1&&engine->AAM_exp->isAdaptive)||(startNum%5==3&&!engine->AAM_exp->isAdaptive))
		else
		{
			//cout<<"begin update\n";
			if (engine->AAM_exp->isAdaptive&&curStatus==1)
			{
				//GTB("START");


				float tmp;
				curStatus=iterate_combination(0,0,0,0,tmp,NULL,false,false,true);


				/*GTE("START");

				gCodeTimer.printTimeTree();
				double time = total_fps;
				cout<<"used time per iteration: "<<time<<endl;*/

				//isSecceed=engine->track_combine(colorIMG_Gray,depthImageWarped,curStatus,startX,endX,startY,endY,startNum>0);
				////isSecceed=engine->track_combine(colorIMG_Gray,depthImageWarped,startX,endX,startY,endY,startNum>0);
				//if (initial&&isSecceed)
				//{
				//	initial=false;
				//}

				//if (!isSecceed)//reset all the parameters to initial status
				//{
				//	initial=true;
				//	startNum=0;
				//	engine->hasVelocity=false;
				//}
				//curStatus=0;
				//cout<<"model updated\n";
			}

			//curStatus=0;
			isDrop=true;

		}
		/*GTE("START");
		gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"used time per iteration: "<<time<<endl;*/
		//}

		//m_pDrawColor->Draw(m_colorRGBX, cDepthWidth * cDepthHeight * cBytesPerPixel);

		if (showDetecton)
		{
			GTE("START");

			gCodeTimer.printTimeTree();
			double time = total_fps;
			cout<<"used time per iteration: "<<time<<endl;
		}


		// cvSetWindowProperty( "Facial Features", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN );  
		if(isSecceed)
		{

			//m_pDrawColor->Draw(m_colorRGBX, cDepthWidth * cDepthHeight * cBytesPerPixel);
			if (engine->state!=2&&startNum>0&&abs(engine->currentShape[76]-lastShape[76])<2&&
				abs(engine->currentShape[76+cptsNum]-lastShape[76+cptsNum])<2&&!engine->showNN)
				//if(0)
			{
				float *curShape=engine->currentShape;

				for (int i=0;i<cptsNum;i++)
				{
					float tmp[2];
					tmp[0]=(curShape[i]+lastShape[i])/2;
					tmp[1]=(curShape[i+cptsNum]+lastShape[i+cptsNum])/2;
					/*	if (abs(tmp[0]-lastShape[i])<1&&abs(tmp[1]-lastShape[i+cptsNum])<1)
					{
					currentShape[i]=lastShape[i];
					currentShape[i+cptsNum]=lastShape[i+cptsNum];
					}
					else*/
					{
						currentShape[i]=tmp[0];
						currentShape[i+cptsNum]=tmp[1];
					}

				}
			}
			else
			{
				//cout<<startNum<<" not smoothing\n";
				if (engine->state==2)
				{
					for (int i=0;i<cptsNum*2;i++)
					{
						currentShape[i]=engine->currentDetection[i];
					}

					if (isRecording)
					{
						for (int i=0;i<cptsNum*2;i++)
						{
							ptsList.push_back(abs(currentShape[i]-engine->currentShape[i]));
						}
					}
				}
				else
				{
					for (int i=0;i<cptsNum*2;i++)
					{
						currentShape[i]=engine->currentShape[i];
					}
				}

			}

			//Mat tmpImg;
			//	flip(colorImage,tmpImg,1);
			m_pDrawColor->Draw(colorImage.ptr<BYTE>(), cDepthWidth * cDepthHeight * cBytesPerPixel);

			/*namedWindow("Facial Features",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
			for (int i=0;i<cptsNum;i++)
			{
			circle(colorImageFlip,Point(640-1-currentShape[i]+0.5,currentShape[i+cptsNum]+0.5),1,Scalar(0,255,0));
			}
			imshow("Facial Features",colorImageFlip(Range(153,289),Range(205,348)));*/

			for (int i=0;i<cptsNum*2;i++)
			{
				lastShape[i]=engine->currentShape[i];
			}
		}
		else if (!isSecceed&&!isDrop)
		{
			//Mat tmpImg;
			//flip(colorImage,tmpImg,1);
			m_pDrawColor->DrawImgOnly(colorImage.ptr<BYTE>(), cDepthWidth * cDepthHeight * cBytesPerPixel);
			//namedWindow("Facial Features",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
			//flip(colorImage,colorImage,1);
			/*for (int i=0;i<cptsNum*2;i++)
			currentShape[i]=lastShape[i];
			m_pDrawColor->Draw(colorImage.ptr<BYTE>(), cDepthWidth * cDepthHeight * cBytesPerPixel);*/
			//namedWindow("Facial Features",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
			//for (int i=0;i<cptsNum;i++)
			//{
			//	circle(colorImageFlip,Point(640-1-engine->currentShape[i]+0.5,engine->currentShape[i+cptsNum]+0.5),1,Scalar(0,255,0));
			//}
			////imshow("Facial Features",colorImageFlip);
			//imshow("Facial Features",colorImageFlip(Range(60,360),Range(200,500)));
			//imwrite("Peizhao.jpg",colorImageFlip);;
		}
		//else if (isDrop)
		//{
		//	for (int i=0;i<cptsNum;i++)

		//	{
		//		circle(colorImageFlip,Point(640-1-engine->currentShape[i],engine->currentShape[i+cptsNum]),1,Scalar(0,255,0));
		//	}
		//	//flip(colorImage,colorImage,1);
		//	imshow("Facial Features",colorImageFlip);
		//}

		//imshow("Facial Features",colorImage(Range(60,360),Range(200,500)));
		//	waitKey(1);
		//GTE("START");
		////QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 


		//imshow("Facial Features",colorImage(Range(60,360),Range(200,500)));
		//waitKey(1);



		//double time=(t2-t1)*1000/persecond; 

		/*gCodeTimer.printTimeTree();
		double time = total_fps;
		int fontFace = FONT_HERSHEY_PLAIN;
		char name_withSize1[50];
		sprintf(name_withSize1, "%f",(float)(time));
		putText(colorImage,name_withSize1,Point(250,80),fontFace,1.2,Scalar(0,255,0));

		namedWindow("FPS",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
		imshow("FPS",colorImage(Range(60,360),Range(200,500)));
		waitKey(1);*/

		//GTE("START");

		/*gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"used time per iteration: "<<time<<"  /60= "<<time/60<<endl;*/

		//if (startNum<=30)
		{
			startNum++;
		}
		/*	if (usingOriginDepth)
		engine->track_combine_autoStatus(colorIMG_Gray,depthImage);
		else
		engine->track_combine_autoStatus(colorIMG_Gray,depthImageWarped);*/
	}
	else
	{
		//namedWindow("Facial Features",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
		//flip(colorImage,colorImage,1);
		/*	namedWindow("Facial Features",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
		imshow("Facial Features",colorImageFlip(Range(153,289),Range(205,348)));*/

		for (int i=0;i<engine->ptsNum*2;i++)
			currentShape[i]=lastShape[i];

		//Mat tmpImg;
		//flip(colorImage,tmpImg,1);
		m_pDrawColor->Draw(colorImage.ptr<BYTE>(), cDepthWidth * cDepthHeight * cBytesPerPixel);
		//namedWindow("Facial Features",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
		//putText(colorImage,"123456",Point(250,80),FONT_HERSHEY_PLAIN,1.2,Scalar(0,255,0));
		//// cvSetWindowProperty( "Facial Features", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN );  
		//imshow("Facial Features",colorImage);
		//waitKey(30);
	}
	//if (showDetecton)
	//{
	//	// loop over each row and column of the color
	//	for (LONG y = 0; y < cColorHeight; ++y)
	//	{
	//		//LONG* pDest = (LONG*)((BYTE*)msT.pData + msT.RowPitch * y);
	//		for (LONG x = 0; x < cColorWidth; ++x)
	//		{
	//			// calculate index into color array
	//			LONG colorIndex = colorInDepthX + colorInDepthY * cColorWidth;
	//			depthImage.at<float>(y,x)=
	//		}
	//	}
	//save depth and color image
	//save color image



	//double testMaxval=500;
	//if (!showOriginalImage&&cframeId>30&&!showDetecton)
	//{
	//	 namedWindow("Current frame");
	//	int maxIdx[3]; 
	//	minMaxIdx(depthImageWarped, 0, &testMaxval, 0, maxIdx);
	//	imshow("Current frame",depthImageWarped/testMaxval);

	//	//// Retrieve the path to My Photos
	//	WCHAR screenshotPath[MAX_PATH];
	//	StringCchPrintfW(screenshotPath,  _countof(screenshotPath), L"G:\\face database\\kinect data\\images\\Fuhao_%d.xml", cframeId);

	//	char strDes[MAX_PATH];
	//	wcstombs(strDes, screenshotPath, MAX_PATH);
	//	/*CvMat tmpDepth=depthImageWarped;
	//	cvSave(strDes, &tmpDepth); */

	//	/*imwrite(strDes,depthImageWarped/testMaxval*255.0f);*/
	//	//FileStorage fs(strDes, FileStorage::WRITE); 
	//	//fs  << depthImageWarped; 

	//	//imwrite(strDes,depthImageWarped);

	//	//StringCchPrintfW(screenshotPath,  _countof(screenshotPath), L"G:\\face database\\kinect data\\images\\Fuhao_%d.jpg", cframeId);
	//	//wcstombs(strDes, screenshotPath, MAX_PATH);
	//	//imwrite(strDes,colorImage);
	//}
	//else
	//{
	//	//namedWindow("Current frame");
	////	namedWindow("Color");
	////	imshow("Current frame",depthImage);
	//}

	//namedWindow("Original depth");
	//imshow("Original depth",depthImage/testMaxval);
	// Draw the data with Direct2D
	//if(!showDetecton)
	//{
	//	namedWindow("Facial Features",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
	//	// cvSetWindowProperty( "Facial Features", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN );  
	//	imshow("Facial Features",colorImage);
	//	waitKey(1);
	//}


	//m_pDrawDepth->Draw(m_colorRGBX, cDepthWidth * cDepthHeight * cBytesPerPixel);

	depthImageWarped.release();
}

HRESULT CColorDepthBasics::ProcessDepth()
{
	NUI_IMAGE_FRAME imageFrame;

	HRESULT hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pDepthStreamHandle, 0, &imageFrame);
	if ( FAILED(hr) ) { return hr; }

	INuiFrameTexture * pTexture = imageFrame.pFrameTexture;

	NUI_LOCKED_RECT LockedRect;
	hr = imageFrame.pFrameTexture->LockRect(0, &LockedRect, NULL, 0);
	if ( FAILED(hr) ) { return hr; }


	memcpy(m_depthD16, LockedRect.pBits, LockedRect.size);
	// Make sure we've received valid data
	if (LockedRect.Pitch != 0)
	{
		//BYTE * rgbrun = m_depthRGBX;
		//const USHORT * pBufferRun = (const USHORT *)LockedRect.pBits;

		//// end pixel is start + width*height - 1
		//const USHORT * pBufferEnd = pBufferRun + (cDepthWidth * cDepthHeight);

		//while ( pBufferRun < pBufferEnd )
		//{
		//	// discard the portion of the depth that contains only the player index
		//	USHORT depth = NuiDepthPixelToDepth(*pBufferRun);

		//	// to convert to a byte we're looking at only the lower 8 bits
		//	// by discarding the most significant rather than least significant data
		//	// we're preserving detail, although the intensity will "wrap"
		//	BYTE intensity = static_cast<BYTE>(depth % 256);

		//	// Write out blue byte
		//	*(rgbrun++) = intensity;

		//	// Write out green byte
		//	*(rgbrun++) = intensity;

		//	// Write out red byte
		//	*(rgbrun++) = intensity;

		//	// We're outputting BGR, the last byte in the 32 bits is unused so skip it
		//	// If we were outputting BGRA, we would write alpha here.
		//	++rgbrun;

		//	// Increment our index into the Kinect's depth buffer
		//	++pBufferRun;
		//}

		// Draw the data with Direct2D
		/*	if(showOriginalImage)
		m_pDrawDepth->Draw(m_depthRGBX, cDepthWidth * cDepthHeight * cBytesPerPixel);*/

		m_bDepthReceived = true;
	}
	hr = imageFrame.pFrameTexture->UnlockRect(0);
	if ( FAILED(hr) ) { return hr; };




	hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_pDepthStreamHandle, &imageFrame);
	// copy to our d3d 11 depth texture
	return hr;


	//HRESULT hr;
	//NUI_IMAGE_FRAME imageFrame;

	//// Attempt to get the depth frame
	//hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pDepthStreamHandle, 0, &imageFrame);
	//if (FAILED(hr))
	//{
	//	return;
	//}

	//INuiFrameTexture * pTexture = imageFrame.pFrameTexture;
	//NUI_LOCKED_RECT LockedRect;

	//// Lock the frame data so the Kinect knows not to modify it while we're reading it
	//pTexture->LockRect(0, &LockedRect, NULL, 0);

	//// Make sure we've received valid data
	//if (LockedRect.Pitch != 0)
	//{
	//	BYTE * rgbrun = m_depthRGBX;
	//	const USHORT * pBufferRun = (const USHORT *)LockedRect.pBits;

	//	// end pixel is start + width*height - 1
	//	const USHORT * pBufferEnd = pBufferRun + (cDepthWidth * cDepthHeight);

	//	while ( pBufferRun < pBufferEnd )
	//	{
	//		// discard the portion of the depth that contains only the player index
	//		USHORT depth = NuiDepthPixelToDepth(*pBufferRun);

	//		// to convert to a byte we're looking at only the lower 8 bits
	//		// by discarding the most significant rather than least significant data
	//		// we're preserving detail, although the intensity will "wrap"
	//		BYTE intensity = static_cast<BYTE>(depth % 256);

	//		// Write out blue byte
	//		*(rgbrun++) = intensity;

	//		// Write out green byte
	//		*(rgbrun++) = intensity;

	//		// Write out red byte
	//		*(rgbrun++) = intensity;

	//		// We're outputting BGR, the last byte in the 32 bits is unused so skip it
	//		// If we were outputting BGRA, we would write alpha here.
	//		++rgbrun;

	//		// Increment our index into the Kinect's depth buffer
	//		++pBufferRun;
	//	}

	//	// Draw the data with Direct2D
	//	m_pDrawDepth->Draw(m_depthRGBX, cDepthWidth * cDepthHeight * cBytesPerPixel);
	//}

	//// We're done with the texture so unlock it
	//pTexture->UnlockRect(0);

	//// Release the frame
	//m_pNuiSensor->NuiImageStreamReleaseFrame(m_pDepthStreamHandle, &imageFrame);
}

/// <summary>
/// Handle new color data
/// </summary>
/// <returns>indicates success or failure</returns>
HRESULT CColorDepthBasics::ProcessColor()
{

	NUI_IMAGE_FRAME imageFrame;

	HRESULT hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pColorStreamHandle, 0, &imageFrame);
	if ( FAILED(hr) ) { return hr; }

	INuiFrameTexture * pTexture = imageFrame.pFrameTexture;
	NUI_LOCKED_RECT LockedRect;
	hr = imageFrame.pFrameTexture->LockRect(0, &LockedRect, NULL, 0);
	if ( FAILED(hr) ) { return hr; }

	//if(showOriginalImage)
	//m_pDrawColor->Draw(static_cast<BYTE *>(LockedRect.pBits), LockedRect.size);
	//memcpy(m_colorRGBX, LockedRect.pBits, LockedRect.size);	
	//BYTE *tmp=colorImage.ptr<BYTE>();
	//cout<<colorImage.rows<<" "<<colorImage.cols<<" "<<LockedRect.size/sizeof(BYTE)<<endl;

	memcpy(colorImage.ptr<BYTE>(), LockedRect.pBits, LockedRect.size);	
	flip(colorImage,colorImageFlip,1);

	m_bColorReceived = true;

	hr = imageFrame.pFrameTexture->UnlockRect(0);


	if ( FAILED(hr) ) { return hr; };

	hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_pColorStreamHandle, &imageFrame);

	return hr;


	//HRESULT hr;
	//NUI_IMAGE_FRAME imageFrame;

	//// Attempt to get the color frame
	//hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pColorStreamHandle, 0, &imageFrame);
	//if (FAILED(hr))
	//{
	//	return;
	//}

	//INuiFrameTexture * pTexture = imageFrame.pFrameTexture;
	//NUI_LOCKED_RECT LockedRect;

	//// Lock the frame data so the Kinect knows not to modify it while we're reading it
	//pTexture->LockRect(0, &LockedRect, NULL, 0);

	//// Make sure we've received valid data
	//if (LockedRect.Pitch != 0)
	//{
	//	// Draw the data with Direct2D
	//	m_pDrawColor->Draw(static_cast<BYTE *>(LockedRect.pBits), LockedRect.size);

	//	// If the user pressed the screenshot button, save a screenshot
	//	if (m_bSaveScreenshot)
	//	{
	//		WCHAR statusMessage[cStatusMessageMaxLen];

	//		// Retrieve the path to My Photos
	//		WCHAR screenshotPath[MAX_PATH];
	//		GetScreenshotFileName(screenshotPath, _countof(screenshotPath));

	//		// Write out the bitmap to disk
	//		hr = SaveBitmapToFile(static_cast<BYTE *>(LockedRect.pBits), cColorWidth, cColorHeight, 32, screenshotPath);

	//		if (SUCCEEDED(hr))
	//		{
	//			// Set the status bar to show where the screenshot was saved
	//			StringCchPrintf( statusMessage, cStatusMessageMaxLen, L"Screenshot saved to %s", screenshotPath);
	//		}
	//		else
	//		{
	//			StringCchPrintf( statusMessage, cStatusMessageMaxLen, L"Failed to write screenshot to %s", screenshotPath);
	//		}

	//		SetStatusMessage(statusMessage);

	//		// toggle off so we don't save a screenshot again next frame
	//		m_bSaveScreenshot = false;
	//	}
	//}

	//// We're done with the texture so unlock it
	//pTexture->UnlockRect(0);

	//// Release the frame
	//m_pNuiSensor->NuiImageStreamReleaseFrame(m_pColorStreamHandle, &imageFrame);
}


/// <summary>
/// Set the status bar message
/// </summary>
/// <param name="szMessage">message to display</param>
void CColorDepthBasics::SetStatusMessage(WCHAR * szMessage)
{
	SendDlgItemMessageW(m_hWnd, IDC_STATUS, WM_SETTEXT, 0, (LPARAM)szMessage);
}

/// <summary>
/// Save passed in image data to disk as a bitmap
/// </summary>
/// <param name="pBitmapBits">image data to save</param>
/// <param name="lWidth">width (in pixels) of input image data</param>
/// <param name="lHeight">height (in pixels) of input image data</param>
/// <param name="wBitsPerPixel">bits per pixel of image data</param>
/// <param name="lpszFilePath">full file path to output bitmap to</param>
/// <returns>indicates success or failure</returns>
HRESULT CColorDepthBasics::SaveBitmapToFile(BYTE* pBitmapBits, LONG lWidth, LONG lHeight, WORD wBitsPerPixel, LPCWSTR lpszFilePath)
{
	DWORD dwByteCount = lWidth * lHeight * (wBitsPerPixel / 8);

	BITMAPINFOHEADER bmpInfoHeader = {0};

	bmpInfoHeader.biSize        = sizeof(BITMAPINFOHEADER);  // Size of the header
	bmpInfoHeader.biBitCount    = wBitsPerPixel;             // Bit count
	bmpInfoHeader.biCompression = BI_RGB;                    // Standard RGB, no compression
	bmpInfoHeader.biWidth       = lWidth;                    // Width in pixels
	bmpInfoHeader.biHeight      = -lHeight;                  // Height in pixels, negative indicates it's stored right-side-up
	bmpInfoHeader.biPlanes      = 1;                         // Default
	bmpInfoHeader.biSizeImage   = dwByteCount;               // Image size in bytes

	BITMAPFILEHEADER bfh = {0};

	bfh.bfType    = 0x4D42;                                           // 'M''B', indicates bitmap
	bfh.bfOffBits = bmpInfoHeader.biSize + sizeof(BITMAPFILEHEADER);  // Offset to the start of pixel data
	bfh.bfSize    = bfh.bfOffBits + bmpInfoHeader.biSizeImage;        // Size of image + headers

	// Create the file on disk to write to
	HANDLE hFile = CreateFileW(lpszFilePath, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	// Return if error opening file
	if (NULL == hFile) 
	{
		return E_ACCESSDENIED;
	}

	DWORD dwBytesWritten = 0;

	// Write the bitmap file header
	if ( !WriteFile(hFile, &bfh, sizeof(bfh), &dwBytesWritten, NULL) )
	{
		CloseHandle(hFile);
		return E_FAIL;
	}

	// Write the bitmap info header
	if ( !WriteFile(hFile, &bmpInfoHeader, sizeof(bmpInfoHeader), &dwBytesWritten, NULL) )
	{
		CloseHandle(hFile);
		return E_FAIL;
	}

	// Write the RGB Data
	if ( !WriteFile(hFile, pBitmapBits, bmpInfoHeader.biSizeImage, &dwBytesWritten, NULL) )
	{
		CloseHandle(hFile);
		return E_FAIL;
	}    

	// Close the file
	CloseHandle(hFile);
	return S_OK;
}