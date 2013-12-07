//------------------------------------------------------------------------------
// <copyright file="ColorBasics.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#pragma once

#include "resource.h"
#include "NuiApi.h"
#include "ImageRenderer.h"
//#include "mat.h"
//#pragma comment(lib,"libmx.lib")
//#pragma comment(lib,"libmat.lib")
#include <vector>
#include "AAM_Detection_Combination.h"
#include "CodeTimer.h"

#include <iostream>
using namespace std;

class CColorDepthBasics
{
	static const int        cColorWidth  = 640;
	static const int        cColorHeight = 480;

	static const int        cDepthWidth  = 640;
	static const int        cDepthHeight = 480;

	static const int        cBytesPerPixel = 4;

	static const int        cStatusMessageMaxLen = MAX_PATH*2;

	LONGLONG   t1,t2; 
	LONGLONG   persecond; 
	double time;

public:
	/// <summary>
	/// Constructor
	/// </summary>

	bool haveTable;
	CColorDepthBasics();

	/// <summary>
	/// Destructor
	/// </summary>
	~CColorDepthBasics();

	/// <summary>
	/// Handles window messages, passes most to the class instance to handle
	/// </summary>
	/// <param name="hWnd">window message is for</param>
	/// <param name="uMsg">message</param>
	/// <param name="wParam">message data</param>
	/// <param name="lParam">additional message data</param>
	/// <returns>result of message processing</returns>
	static LRESULT CALLBACK MessageRouter(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

	/// <summary>
	/// Handle windows messages for a class instance
	/// </summary>
	/// <param name="hWnd">window message is for</param>
	/// <param name="uMsg">message</param>
	/// <param name="wParam">message data</param>
	/// <param name="lParam">additional message data</param>
	/// <returns>result of message processing</returns>
	LRESULT CALLBACK        DlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

	/// <summary>
	/// Creates the main window and begins processing
	/// </summary>
	/// <param name="hInstance"></param>
	/// <param name="nCmdShow"></param>
	int                     Run(HINSTANCE hInstance, int nCmdShow);

	LRESULT HandleMessages(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	//////////////////////////////////////////
	//added for randomized trees
	RandTree *rt;
	bool showDetecton;
	Mat colorImage,depthImage;
	Mat colorImageFlip;
	//Mat depthImageWarped;
	float standardDepth;

	AAM_Detection_Combination *engine;
	int startX,endX,startY,endY;
	bool initial;
	int curStatus; //0: tracking,1: ready to update models
	int startNum;
	//////////////////////////////////////////
	bool isRecording;
	vector<float> ptsList;

private:
	HWND                    m_hWnd;

	bool                    m_bSaveScreenshot;

	// Current Kinect
	INuiSensor*             m_pNuiSensor;

	// Direct2D
	ImageRenderer*          m_pDrawColor;

	ImageRenderer*          m_pDrawDepth;

	ID2D1Factory*           m_pD2DFactory;

	HANDLE                  m_pColorStreamHandle;
	HANDLE                  m_hNextColorFrameEvent;

	HANDLE                  m_pDepthStreamHandle;
	HANDLE                  m_hNextDepthFrameEvent;

	BYTE*                   m_depthRGBX;

	/// <summary>
	/// Main processing function
	/// </summary>
	void                    Update();

	/// <summary>
	/// Create the first connected Kinect found 
	/// </summary>
	/// <returns>S_OK on success, otherwise failure code</returns>
	HRESULT                 CreateFirstConnected();

	/// <summary>
	/// Handle new color data
	/// </summary>
	HRESULT                    ProcessColor();

	HRESULT                    ProcessDepth();

	void                    ProcessDepthwithMapping();
	float lastShape[200],currentShape[200];

	// for mapping depth to color
	USHORT*                             m_depthD16;
	HRESULT					   MapColorToDepth();
	 LONG*                               m_colorCoordinates;
	 BYTE*                               m_colorRGBX;
	 BYTE*                               m_colorRGBXAligned;

	 float *depthData;
	 vector<float> *depthData1;
	 int totalFrameNo;
	 long m_colorToDepthDivisor;

	 int frameId;
	 int startInd,cframeId,endInd;
	 int realStartInd;

	bool m_bDepthReceived,m_bColorReceived;

	bool outputVideoOrigin,outputVideoAligned,outputDepth;

	bool outputDepthWarped;
	bool usingOriginDepth;
	Mat *depthImagesWarped;
	

	vector<BYTE *>videoOrign,videoAligned;


	void writeMat(float *depthInfo,WCHAR *path,int id=-1);

	bool showOriginalImage;
	bool capture;
	/// <summary>
	/// Set the status bar message
	/// </summary>
	/// <param name="szMessage">message to display</param>
	void                    SetStatusMessage(WCHAR* szMessage);

	/// <summary>
	/// Save passed in image data to disk as a bitmap
	/// </summary>
	/// <param name="pBitmapBits">image data to save</param>
	/// <param name="lWidth">width (in pixels) of input image data</param>
	/// <param name="lHeight">height (in pixels) of input image data</param>
	/// <param name="wBitsPerPixel">bits per pixel of image data</param>
	/// <param name="lpszFilePath">full file path to output bitmap to</param>
	/// <returns>S_OK on success, otherwise failure code</returns>
	HRESULT                 SaveBitmapToFile(BYTE* pBitmapBits, LONG lWidth, LONG lHeight, WORD wBitsPerPixel, LPCTSTR lpszFilePath);


};
