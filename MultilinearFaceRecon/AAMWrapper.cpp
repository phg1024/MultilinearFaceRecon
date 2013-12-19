#include "AAMWrapper.h"
#include "Utils/Timer.h"


AAMWrapper::AAMWrapper(void)
{
	PhGUtils::Timer t;
	t.tic();
	setup();
	t.toc("AAM setup");
}


AAMWrapper::~AAMWrapper(void)
{
	delete engine;
}

void AAMWrapper::setup() {
	string datapath = "C:\\Users\\Peihong\\Desktop\\Data\\Fuhao\\model\\";
	string searchPicDir;
	string savePrefix;
	string AAMSearchPrefix;
	string colorRT_model;
	string depthRT_model;
	string AAMModelPath;
	string alignedShapeDir;

	searchPicDir=datapath + "imgList.txt";
	savePrefix="DougTalking_AAM_Sin";

	colorRT_model= datapath + "trainedTree_15_12_56_22_1.txt";
	depthRT_model= datapath + "trainedTree_15_12_64_22_0.txt";

	AAMModelPath= datapath + "trainedResult_90_90.txt";
	alignedShapeDir= datapath + "allignedshape_90_90.txt";

	engine=new AAM_Detection_Combination(0.9,0.05,0.003,0,colorRT_model,depthRT_model,AAMModelPath,alignedShapeDir,true);

	// initialize the wrapper
	initial = true;
	curStatus = 0;
	startNum = 0;
	isRecording = false;

	f.reserve(256);
}

const vector<float>& AAMWrapper::track( const unsigned char* cimg, const unsigned char* dimg, int w, int h )
{

		/// realtime tracking related

		// copy the image over here
		Mat colorImage;
		colorImage.create(h, w, CV_8UC4);
		memcpy(colorImage.ptr<BYTE>(), cimg, sizeof(unsigned char)*w*h*4);

		// convert to gray image
		Mat colorIMG_Gray;
		cvtColor(colorImage, colorIMG_Gray, CV_BGR2GRAY);

		Mat depthImg = cv::Mat::zeros(h, w, CV_32FC1);
		const float standardDepth = 750.0;
		for (int i=0, idx=0;i<depthImg.rows;i++)
		{
			for (int j=0;j<depthImg.cols;j++,idx+=4)
			{
				int tmp=(dimg[idx]<<16|dimg[idx+1]<<8|dimg[idx+2]<<0);
				depthImg.at<float>(i,j)=tmp/standardDepth;
			}
		}

		medianBlur(depthImg,depthImg,3);

		/*
		// for debug
		namedWindow("1",0);
		imshow("1",depthImageWarped);
		*/

		int cptsNum=engine->AAM_exp->meanShape->ptsNum;
		if (initial)
		{
			startX=w-348;
			endX=w-205;
			startY=153;
			endY=289;
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

		if (startX<50||startY<50||endX>w-50||endY>h-50)
		{
			//reset parameters
			initial=true;
			startNum=0;
			engine->hasVelocity=false;
		}

		bool isSecceed=false;
		bool isDrop=false;
		//if(1)
		if (((curStatus==0&&engine->AAM_exp->isAdaptive)&&(startNum%5!=4))||(!engine->AAM_exp->isAdaptive&&startNum%5!=4))
			//if (startNum%5!=3)
		{
			//	GTB("START");
			isSecceed=engine->track_combine(colorIMG_Gray,depthImg,curStatus,startX,endX,startY,endY,startNum>0);
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
		}
		else
		{
			//cout<<"begin update\n";
			if (engine->AAM_exp->isAdaptive&&curStatus==1)
			{
				float tmp;
				curStatus=iterate_combination(0,0,0,0,tmp,NULL,false,false,true);
				//cout<<"model updated\n";
			}

			//curStatus=0;
			isDrop=true;

		}

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
			//flip(colorImage,tmpImg,1);
			//m_pDrawColor->Draw(tmpImg.ptr<BYTE>(), cDepthWidth * cDepthHeight * cBytesPerPixel);

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
			//m_pDrawColor->DrawImgOnly(tmpImg.ptr<BYTE>(), cDepthWidth * cDepthHeight * cBytesPerPixel);
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

		if( isSecceed )		
		{
			cout << "Succeeded." << endl;
			f.assign(lastShape, lastShape + cptsNum*2);
			return f;
		}
		else
		{
			cout << "Failed." << endl;
			return eptf;
		}
}

void AAMWrapper::reset()
{
	initial = true;
}
