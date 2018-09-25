#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
#define detectOK 1;

//计算每个小波处理单元的小波系数（默认2 * 2）
int haarTrans(Mat haarBlock, Mat& haarMat)
{
	int rows = haarBlock.rows;
	int cols = haarBlock.cols;
	Mat rowTrans = Mat::zeros(Size(2, 2), CV_64FC1);   //行变换的结果
	Mat haarTrans = Mat::zeros(Size(2, 2), CV_64FC1);  //小波变换的最终结果
	for (int i = 0;i < rows / 2;i++)
	{
		for (int j = 0;j < cols;j++)
		{
			rowTrans.at<double>(i, j) = (haarBlock.at<double>(2 * i, j) +
				haarBlock.at<double>(2 * i + 1, j)) / 2.0;
			rowTrans.at<double>(i + rows / 2, j) = (haarBlock.at<double>(2 * i, j) -
				haarBlock.at<double>(2 * i + 1, j)) / 2.0;
		}
	}

	for (int i = 0;i < rows;i++)
	{
		for (int j = 0;j < cols / 2;j++)
		{
			haarTrans.at<double>(i, j) = (rowTrans.at<double>(i, 2 * j) +
				rowTrans.at<double>(i, 2 * j + 1)) / 2.0;
			haarTrans.at<double>(i, j + cols / 2) = (rowTrans.at<double>(i, 2 * j) -
				rowTrans.at<double>(i, 2 * j + 1)) / 2.0;
		}
	}

	//将小波处理单元的Haar小波系数存入数组中
	haarMat.at<double>(0, 0) = haarTrans.at<double>(0, 0);
	haarMat.at<double>(0, 1) = (haarTrans.at<double>(1, 0) + haarTrans.at<double>(0, 1)
		+ haarTrans.at<double>(1, 1)) / 3.0;

	return detectOK;
}

/*计算每个统计单元小波系数的均值和方差*/
int calcStaValue(Mat staUnit, Mat& staUnitMean, Mat& staUnitVar)
{
	Mat staUnitSum = Mat::zeros(Size(6, 1), CV_64FC1);  //每个统计单元的各小波系数和
	Mat staUnitValue = Mat::zeros(Size(6, 4), CV_64FC1);  //四个小波统计单元的系数
	Mat haarMat = Mat::zeros(Size(2, 1), CV_64FC1);  //存放每个小波变换单元的四个系数
	//将统计单元的RGB通道进行分解
	vector<Mat> unitChannels;
	split(staUnit, unitChannels);
	Mat singleStaUnit(staUnit.size(), CV_64FC1);
	for (int n = 0;n < 3;n++)
	{
		singleStaUnit = unitChannels[n];
		for (int i = 0;i < 2;i++)
		{
			for (int j = 0;j < 2;j++)
			{
				Rect staUnitRect(2 * j, 2 * i, 2, 2);
				Mat haarBlock = singleStaUnit(staUnitRect);  //小波变换单元
				haarTrans(haarBlock, haarMat);  //求出小波变换单元的两个系数
				//将每个小波单元的系数存到统计单元的系数矩阵中
				staUnitValue.at<double>(2 * i + j, 2 * n) = haarMat.at<double>(0, 0);
				staUnitValue.at<double>(2 * i + j, 2 * n + 1) = haarMat.at<double>(0, 1);
				//求出统计单元内各系数的和
				staUnitSum(Rect(2 * n, 0, 2, 1)) += haarMat.clone();
			}
		}
	}

	//计算统计单元内的小波系数均值
	staUnitMean = staUnitSum.mul(0.25);
	//计算统计单元内小波系数的方差
	for (int i = 0;i < 6;i++)
	{
		for (int j = 0;j < 6;j++)
		{
			//计算第i和第j个小波系数的方差
			double haarUnitVarSum = 0;
			for (int k = 0;k < 4;k++)
			{
				haarUnitVarSum += (staUnitValue.at<double>(k, i) - staUnitMean.at<double>(0, i))
					*(staUnitValue.at<double>(k, j) - staUnitMean.at<double>(0, j));
			}
			staUnitVar.at<double>(i, j) = haarUnitVarSum / 3.0;
		}
	}

	return detectOK;
}


double gamma(double xx)
{
	double coef_const[7];
	double step = 2.50662827465;
	double HALF = 0.5;
	double ONE = 1;
	double FPF = 5.5;
	double SER, temp, x, y;
	int j;

	coef_const[1] = 76.18009173;
	coef_const[2] = -86.50532033;
	coef_const[3] = 24.01409822;
	coef_const[4] = -1.231739516;
	coef_const[5] = 0.00120858003;
	coef_const[6] = -0.00000536382;

	x = xx - ONE;
	temp = x + FPF;
	temp = (x + HALF)*log(temp) - temp;
	SER = ONE;
	for (j = 1;j <= 6;j++)
	{
		x = x + ONE;
		SER = SER + coef_const[j] / x;
	}
	y = temp + log(step*SER);

	return exp(y);
}


double beta_cf(double a, double b, double x)
{
	int count, count_max = 100;
	double eps = 0.0000001;
	double AM = 1;
	double BM = 1;
	double AZ = 1;
	double QAB;
	double QAP;
	double QAM;
	double BZ, EM, TEM, D, AP, BP, AAP, BPP, AOLD;

	QAB = a + b;
	QAP = a + 1;
	QAM = a - 1;
	BZ = 1 - QAB*x / QAP;

	for (count = 1;count <= count_max;count++)
	{
		EM = count;
		TEM = EM + EM;
		D = EM*(b - count)*x / ((QAM + TEM)*(a + TEM));
		AP = AZ + D*AM;
		BP = BZ + D*BM;
		D = -(a + EM)*(QAB + EM)*x / ((a + TEM)*(QAP + TEM));
		AAP = AP + D*AZ;
		BPP = BP + D*BZ;
		AOLD = AZ;
		AM = AP / BPP;
		BM = BP / BPP;
		AZ = AAP / BPP;
		BZ = 1;
		if (fabs(AZ - AOLD)<eps*fabs(AZ)) 
			return(AZ);
	}
	return AZ;
}


/*F分布概率密度函数*/
double FPdf(double F,double freeDegUp,double freeDegDown)
{
	double FResUp = gamma((freeDegUp + freeDegDown) / 2.0)*pow(freeDegUp, freeDegUp / 2.0)*
		pow(freeDegDown, freeDegDown / 2.0)*pow(F, freeDegUp / 2.0 - 1);
	double FResDown = gamma(freeDegUp / 2.0)*gamma(freeDegDown / 2.0)*
		pow(freeDegUp*F + freeDegDown, (freeDegUp + freeDegDown) / 2.0);
	return FResUp / FResDown;
}


/*F分布的累计分布函数*/
double FCdf(double F, double freeDegUp, double freeDegDown)
{
	double delta = 0.1;

	double fCdfRes = 0;  //该点的累计概率
	for (double i = delta; i < F; i += delta)
	{
		fCdfRes += FPdf(i, freeDegUp, freeDegDown)*delta;
	}
	return fCdfRes;
}

/**********************************************/
double betainc(double x, double a, double b)/* 不完全Beta函数 */
{
	double y, BT, logProp;

	if (x == 0 || x == 1)
		BT = 0;
	else
	{
		logProp = log(gamma(a + b)) - log(gamma(a)) - log(gamma(b));
		BT = exp(logProp + a*log(x) + b*log(1 - x));
	}
	if (x < (a + 1) / (a + b + 2))
		y = BT*beta_cf(a, b, x) / a;
	else
		y = 1 - BT*beta_cf(b, a, 1 - x) / b;

	return y;
}

double FDist(double F, double m, double n)
{
	double xx, p;

	if (m <= 0 || n <= 0) p = -1;
	else if (F>0)
	{
		xx = F / (F + n / m);
		p = betainc(xx, m / 2, n / 2);
	}
	return p;
}
/**********************************************/

int main()
{
	Mat srcImg;
	srcImg = imread("defectPic\\timg.jpg",IMREAD_UNCHANGED);

	float scale = 300.0 / srcImg.cols;
	Mat resizeSrc;
	resize(srcImg, resizeSrc, Size(), scale, scale, 1);
	imshow("resizeSrc", resizeSrc);
	
	int rows = srcImg.rows;
	int cols = srcImg.cols;
	int channel = srcImg.channels();

	Mat imgData;
	srcImg.convertTo(imgData, CV_64FC3, 1);

	//计算统计单元行、列方向上的个数（向下取整）
	int rowSerial = rows / 4;
	int colSerial = cols / 4;
	Mat staUnit = Mat::zeros(Size(4, 4), CV_64FC1);  //统计单元
	Mat singleStaMean = Mat::zeros(Size(6, 1), CV_64FC1);  //单个统计单元的系数均值矩阵
	Mat singleStaVar = Mat::zeros(Size(6, 6), CV_64FC1);  //单个统计单元的系数方差矩阵
	Mat staUnitMean[6];  //所有统计单元的系数均值矩阵
	Mat imgSumMean = Mat::zeros(Size(6, 1), CV_64FC1);  //所有统计单元的各小波系数均值之和
	Mat imgSumVar = Mat::zeros(Size(6, 6), CV_64FC1);  //所有统计单元的各小波系数协方差之和
	Mat imgMeanValue = Mat::zeros(Size(6, 1), CV_64FC1);  //整幅图像的小波系数均值
	Mat imgVarValue = Mat::zeros(Size(6, 6), CV_64FC1);  //整幅图像的小波系数协方差矩阵
	//初始化均值矩阵
	for (int k = 0;k < 6;k++)
	{
		staUnitMean[k] = Mat::zeros(Size(colSerial, rowSerial), CV_64FC1);
	}

	/*求每个小波单元的系数*/
	for (int i = 0;i < rowSerial;i++)
	{
		for (int j = 0;j < colSerial;j++)
		{
			//提取统计单元的四个特征值
			Rect staUnitRoi(4 * j, 4 * i, 4, 4);
			staUnit = imgData(staUnitRoi);
			calcStaValue(staUnit, singleStaMean, singleStaVar);
			for (int k = 0;k < 6;k++)
			{
				staUnitMean[k].at<double>(i, j) = singleStaMean.at<double>(0, k);
			}
			//计算所有统计单元系数均值和方差的和
			imgSumMean += singleStaMean;
			imgSumVar += singleStaVar;
		}
	}

	//计算整幅图像的小波系数均值和协方差
	double unitScale = 1.0*rowSerial*colSerial;
	imgMeanValue = imgSumMean.mul(1.0 / unitScale);
	imgVarValue = imgSumVar.mul(1.0 / unitScale);
	Mat imgVarInv = Mat::ones(Size(6, 6), CV_64FC1)/imgVarValue;
	//cout << "imgMeanValue=" << imgMeanValue << endl;
	//cout << "imgVarValue=" << imgVarValue << endl;
	//计算统计量上限
	int freeDeg1 = 4;  //F分布的两个自由度
	int freeDeg2 = 4 * 4 - 4 - 4 + 1;
	//计算每个统计单元的HotellingT ^ 2统计量
	Mat meanDif = Mat::zeros(Size(6, 1), CV_64FC1);  //区域均值与图像均值的差
	Mat hotelRes;  //每个单元的Hotelling T^2统计值
	Mat defectImg = srcImg.clone();  //用于显示瑕疵的图像
	for (int i = 0;i < rowSerial;i++)
	{
		for (int j = 0;j < colSerial;j++)
		{
			for (int k = 0;k < 6;k++)
			{
				meanDif.at<double>(0, k) = staUnitMean[k].at<double>(i, j) -
					imgMeanValue.at<double>(0, k);
				//cout << meanDif.at<double>(0, k) << " ";
			}
			hotelRes = meanDif*imgVarValue.inv()*(meanDif.t())*(2 * 2);
			//hotelRes = meanDif*imgVarInv*(meanDif.t())*(2 * 2);
			double res = hotelRes.at<double>(0, 0);
			double P1 = 1 - FCdf(res, freeDeg1, freeDeg2);
			double P2 = 1 - FDist(res, freeDeg1, freeDeg2);
			if (P2 < 1e-10)
			{
				Rect rect(4 * j, 4 * i, 4, 4);
				rectangle(defectImg, rect, Scalar(0, 0, 255), 1);
				//cout << "  *******DefectInfo!*******  ";
			}
			//cout << "Res=" << res << endl;
		}
	}

	Mat resizeDefect;
	resize(defectImg, resizeDefect, Size(), scale, scale, 1);
	imshow("resizeDefect", resizeDefect);

	waitKey(0);
	return 0;
}