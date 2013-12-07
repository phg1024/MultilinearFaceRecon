#include "ShowImgPts.h"
int combineIndListForImage[]={0,4,8,12,16,18,20,22,24,26,28,30,42,45,48,51,56,61,76,35,38};
ShowImgPts::ShowImgPts(ID2D1HwndRenderTarget*   m_pRenderTarget,int _ptsNum)
{
	ptsNum=_ptsNum;
	pRender=m_pRenderTarget;
	 ellipse = new D2D1_ELLIPSE[ptsNum];
	 for (int i=0;i<ptsNum;i++)
	 {
		 ellipse[i].radiusX=ellipse[i].radiusY=1;
	 }

	 const D2D1_COLOR_F color = D2D1::ColorF(1.0f, 1.0f, 0);
	 HRESULT hr = pRender->CreateSolidColorBrush(color, &pBrush);

	 if (FAILED(hr))
	 {
		 return;
	 }

	translation = D2D1::Matrix3x2F::Scale(2,2,D2D1::Point2F(320,240));
	fullIndNum=sizeof(combineIndListForImage)/sizeof(int);
}

void ShowImgPts::drawImgPts(ID2D1Bitmap* m_pBitmap,float *pts,int _ptsNum)
{
	
	int effectiveNum=0;
	for (int i=0;i<_ptsNum;i++)
	{
		ellipse[i].point.x=639-pts[i];
		ellipse[i].point.y=pts[i+_ptsNum];
	}

	/*float center[2]={0,0};
	for (int i=0;i<fullIndNum;i++)
	{
		center[0]+=pts[combineIndListForImage[i]];
		center[1]+=pts[combineIndListForImage[i]+_ptsNum];
	}
	center[0]/=fullIndNum;center[1]/=fullIndNum;*/
	//translation = D2D1::Matrix3x2F::Translation(-center[0],-center[1]);
	//translation = D2D1::Matrix3x2F::Scale(2,2,D2D1::Point2F(center[0],center[1]));
	//translation = D2D1::Matrix3x2F::Scale(2,2,D2D1::Point2F(center[0],center[1]));
	pRender->BeginDraw();
	pRender->SetTransform(translation);
	pRender->DrawBitmap(m_pBitmap);
	for (int i=0;i<_ptsNum;i++)
	{
		pRender->FillEllipse(ellipse[i],pBrush);
	}
	pRender->EndDraw();
}

void ShowImgPts::drawImg(ID2D1Bitmap* m_pBitmap)
{
	
	/*int effectiveNum=0;
	for (int i=0;i<_ptsNum;i++)
	{
		ellipse[i].point.x=pts[i];
		ellipse[i].point.y=pts[i+_ptsNum];
	}

	float center[2]={0,0};
	for (int i=0;i<fullIndNum;i++)
	{
		center[0]+=pts[combineIndListForImage[i]];
		center[1]+=pts[combineIndListForImage[i]+_ptsNum];
	}
	center[0]/=fullIndNum;center[1]/=fullIndNum;*/
	//translation = D2D1::Matrix3x2F::Translation(-center[0],-center[1]);
	//translation = D2D1::Matrix3x2F::Scale(2,2,D2D1::Point2F(center[0],center[1]));
	//translation = D2D1::Matrix3x2F::Scale(2,2,D2D1::Point2F(center[0],center[1]));
	pRender->BeginDraw();
	pRender->SetTransform(translation);
	pRender->DrawBitmap(m_pBitmap);
	/*for (int i=0;i<_ptsNum;i++)
	{
		pRender->FillEllipse(ellipse[i],pBrush);
	}*/
	pRender->EndDraw();
}