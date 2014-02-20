#pragma once
#define NOMINMAX
#include <windows.h>
#include <d2d1.h>
#pragma comment(lib, "d2d1")

class ShowImgPts
{
public:
	ShowImgPts(ID2D1HwndRenderTarget*   m_pRenderTarget,int);
	int ptsNum;
	int fullIndNum;
	ID2D1HwndRenderTarget *pRender;
	ID2D1SolidColorBrush    *pBrush;
	D2D1_ELLIPSE            *ellipse;

	void drawImgPts(ID2D1Bitmap* m_pBitmap,float *pts,int);
	void drawImg(ID2D1Bitmap* m_pBitmap);
	D2D1_MATRIX_3X2_F translation;
};