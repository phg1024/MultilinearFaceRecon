#include "W_train.h"

void print_null(const char *s) {}

W_train::W_train(int _startInd)
{
	bias=1;

	startInd=_startInd;

	showSingleStep=false;
}

void W_train::setPara(parameter &param,double c)
{
		//default
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = c;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	

	param.solver_type=12;
	param.p=0;

	set_print_string_function(&print_null);

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
	
}

void W_train::train_singleC(vector<Shape> &ds,vector<LBFFeature> &trainLBF, Mat &curW, float C)
{

	vector<problem> probs;
	obtainProblems(probs,ds,trainLBF);
	

	//set the problem
	train_batch(probs,curW, C);
}


void W_train::train_multipleC(vector<Shape> &ds,vector<LBFFeature> &trainLBF, vector<Mat> &W_List, vector<float> &C_List)
{
	int cNum=C_List.size();

	vector<problem> probs;
	obtainProblems(probs,ds,trainLBF);

	for(int i=0;i<cNum;i++)
	{
		Mat curW;
		train_batch(probs,curW,C_List[i]);
		W_List[i]=curW.clone();
	}

	//destroy all problems
	for(int i=0;i<probs.size();i++)
	{
		free(probs[i].y);
		if(i==0)
		{
			free(x_space);
			free(probs[i].x);
		}
	}
}

void W_train::train_unit(problem &prob, Mat &W, parameter &param)
{
	model* model_;

	////check
	//for(int i=0;i<10;i++)
	//	cout<<prob.x[0][i].index<<" "<<prob.x[0][i].value<<" ";
	//cout<<endl;

	model_=train(&prob, &param);

	/*for(int i=0;i<10;i++)
		cout<<prob.x[0][i].index<<" "<<prob.x[0][i].value<<" ";
	cout<<endl;*/
	
	if(0)
	{
		cout<<"paraInfo: "<<param.C<<" "<<param.eps<<" "<<param.nr_weight<<" "<<
		param.p<<" "<<param.solver_type<<" "<<param.weight<<" "<<(param.weight_label==NULL)<<endl;

		cout<<prob.l<<" "<<prob.n<<" "<<prob.n<<" "<<prob.bias<<endl;

		for(int i=0;i<prob.l;i++)
			{
				if(i%100==0)
					cout<<prob.y[i]<<" ";
			}
		cout<<endl;

			//check x_space
			for(int i=0;i<1000+prob.l;i++)
			{
				if(i%100==0)
					cout<<"[ "<<x_space[i].index<<","<<x_space[i].value<<" ] ";
			}
			cout<<endl;
			//check x
			for(int i=0;i<prob.l;i++)
			{
				if(i%10==0)
					cout<<"[ "<<prob.x[i][0].index<<","<<prob.x[i][0].value<<" ] ";
			}
			cout<<endl;


		//output model
		int i;
		int nr_feature=model_->nr_feature;
		int n;
		const parameter& param = model_->param;

		if(model_->bias>=0)
			n=nr_feature+1;
		else
			n=nr_feature;
		int w_size = n;

		for(int i=0;i<10;i++)
			cout<<model_->w[i]<<" ";

	}
	//set model to W
	int w_size;
	int nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		w_size=nr_feature+1;
	else
		w_size=nr_feature;
	
	W=Mat::zeros(1,w_size,CV_32FC1);
	for(int i=0;i<w_size;i++)
		W.at<float>(i)=model_->w[i];

	free_and_destroy_model(&model_);
}


void W_train::train_batch(vector<problem> &probs, Mat &W, float C)
{
	parameter param;
	setPara(param,C);

	
	Mat curW;
	//showSingleStep=true;

	//srand(0);
	//train_unit(probs[startInd],curW,param);
	//showSingleStep=false;

	W=Mat::zeros(probs.size(),probs[0].n,CV_32FC1);
	//W.row(startInd)+=curW;

	//return;
	#pragma omp parallel for
	for(int i=0;i<probs.size();i++)
	{
		Mat curW;
		train_unit(probs[i],curW,param);
		W.row(i)+=curW;
	}
}


void W_train::obtainProblems(vector<problem> &probs, vector<Shape> &ds, vector<LBFFeature> &trainLBF)
{
	int probNum=ds[0].n*2;
	probs.resize(probNum);

	obtainProblem(probs[0],ds, 0, trainLBF);
	for(int i=1;i<probNum;i++)
		obtainProblem(probs[i],ds,i,trainLBF,&(probs[0]));
}

void W_train::obtainProblem(problem &prob, vector<Shape> &ds, int ptsInd, vector<LBFFeature> &trainLBF, problem *refProb)
{
	if(refProb==NULL)
	{
		prob.l=ds.size();
		prob.bias=bias;//?check

		

		//y
		prob.y = Malloc(double,prob.l);
		double* curY=prob.y;
		for(int i=0;i<prob.l;i++)
			curY[i]=ds[i].ptsVec.at<float>(ptsInd);

		//x
		//obtain element number
		int elements=0;
		for(int i=0;i<trainLBF.size();i++)
			elements+=trainLBF[i].onesInd.size();
		elements+=prob.l;//bias term
		x_space=Malloc(struct feature_node,elements+prob.l);

		prob.x = Malloc(struct feature_node *,prob.l);
		int curFeatureInd=0;
		for(int i=0;i<prob.l;i++)
		{
			prob.x[i]=&x_space[curFeatureInd];
			for(int j=0;j<trainLBF[i].onesInd.size();j++)
			{
				x_space[curFeatureInd].index=trainLBF[i].onesInd[j]+1;
				x_space[curFeatureInd].value=1;
				curFeatureInd++;
			}
			if(prob.bias >= 0)
				x_space[curFeatureInd++].value = prob.bias;
			x_space[curFeatureInd++].index = -1;
		}
		
		int max_index=trainLBF[0].totalNum;
		if(prob.bias >= 0)
		{
			prob.n=max_index+1;
			for(int i=1;i<prob.l;i++)
				(prob.x[i]-2)->index = prob.n;
			x_space[curFeatureInd-2].index = prob.n;
		}
		else
			prob.n=max_index;

		if(0)
		{
			cout<<prob.l<<" "<<prob.n<<" "<<prob.n<<" "<<prob.bias<<endl;

			////checked
			//for(int i=0;i<prob.l;i++)
			//{
			//	if(i%20==0)
			//		cout<<prob.y[i]<<" ";
			//}

			//check X here. acturally the x_space
			cout<<elements<<endl;

			//check x_space
			for(int i=0;i<elements+prob.l;i++)
			{
				if(i%100==0)
					cout<<"[ "<<x_space[i].index<<","<<x_space[i].value<<" ] ";
			}
			cout<<endl;
			//check x
			for(int i=0;i<prob.l;i++)
			{
				if(i%10==0)
					cout<<"[ "<<prob.x[i][0].index<<","<<prob.x[i][0].value<<" ] ";
			}
			cout<<endl;
		}
	
	}
	else
	{
		prob.l=refProb->l; prob.bias=refProb->bias;prob.n=refProb->n;
		prob.x=refProb->x; 

		prob.y = Malloc(double,prob.l);
		double* curY=prob.y;
		for(int i=0;i<prob.l;i++)
			curY[i]=ds[i].ptsVec.at<float>(ptsInd);
	}

	
}

