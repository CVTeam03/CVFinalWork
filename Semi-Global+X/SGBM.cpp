#include <opencv2/opencv.hpp>  
#include <iostream>  
 
using namespace std;
using namespace cv;

int imageWidth = 0;                              
int imageHeight = 0;
Size imageSize;
 
Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat sobelLimg, sobelRimg;
float*** cost_array;
int *** left_cost;
int *** right_cost;
int *** up_cost;
int *** down_cost;
Mat disparity;

Mat right_disparity;
int *** rightimg_cost;

Mat match_left;
Mat match_right;
Mat final_disparity;
bool ** flags;
Mat final_disparity_after_fill;
vector<pair<int,int>>occlusions;
vector<pair<int,int>>mismatches;
 
 
int minDisparities=2,maxDisparities=64,P1=10,P2=150;
int adjust=true;
int sub_pixel=true;
int uniqueness=true,disparity_threshold=95;
int del_little_area=true,diff_threshold = 5,min_size=32;
int is_fill_blanck_area=true,occlusion_threshold=3,max_range=10;
int is_check_consistence=true,consistence_threshold=3;

void init()
{
    rgbImageL = imread("images/teddy/im2.png");
	cvtColor(rgbImageL, grayImageL, COLOR_BGR2GRAY);
	rgbImageR = imread("images/teddy/im6.png");
	cvtColor(rgbImageR, grayImageR, COLOR_BGR2GRAY);
    Mat kernel_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    filter2D(grayImageL,sobelLimg,CV_16S,kernel_x);
    filter2D(grayImageR,sobelRimg,CV_16S,kernel_x);
    imageWidth = rgbImageL.cols; // 图像宽度
    imageHeight = rgbImageL.rows; // 图像高度
    imageSize = Size(imageWidth, imageHeight);
    disparity = Mat::zeros(imageSize, CV_32FC1);
    match_left = Mat::zeros(imageSize, CV_32FC1);
    match_right = Mat::zeros(imageSize, CV_32FC1);
    cost_array = new float**[imageHeight];
    left_cost = new int**[imageHeight];
    right_cost = new int**[imageHeight];
    up_cost = new int**[imageHeight];
    down_cost = new int**[imageHeight];
    rightimg_cost = new int**[imageHeight];
    right_disparity = Mat::zeros(imageSize, CV_32FC1);
    final_disparity = Mat::zeros(imageSize, CV_32FC1);
    final_disparity_after_fill = Mat::zeros(imageSize, CV_32FC1);
    flags=new bool*[imageHeight];
    occlusions.clear();
    mismatches.clear();
    for(int i=0;i<imageHeight;i++)
    {
        cost_array[i] = new float*[imageWidth];
        left_cost[i] = new int*[imageWidth];
        right_cost[i] = new int*[imageWidth];
        up_cost[i] = new int*[imageWidth];
        down_cost[i] = new int*[imageWidth];
        rightimg_cost[i] = new int*[imageWidth];
        flags[i]=new bool[imageWidth]{};
        for(int j=0;j<imageWidth;j++)
        {
            cost_array[i][j] = new float[64]{};
            left_cost[i][j] = new int[64]{};
            right_cost[i][j] = new int[64]{};
            up_cost[i][j] = new int[64]{};
            down_cost[i][j] = new int[64]{};
            rightimg_cost[i][j] = new int[64]{};
        }
    }
    namedWindow("left", WINDOW_AUTOSIZE);
    imshow("left", rgbImageL);
    namedWindow("right", WINDOW_AUTOSIZE);
    imshow("right", rgbImageR);
}


void compute_cost()
{
    for(int i=1;i<imageHeight-1;i++)
    for(int j=1;j<imageWidth-1;j++)
    {
        for(int d=minDisparities;d<=maxDisparities;d++)
        {
            int cost = 0;
            if(j-d<=0)
            {
                cost = 100000;
            }
            else
            {
                float gray_l_sub_cost=0.5f*(grayImageL.at<uchar>(i,j)+grayImageL.at<uchar>(i,j-1));
                float gray_l_add_cost=0.5f*(grayImageL.at<uchar>(i,j)+grayImageL.at<uchar>(i,j+1));
                float gray_r_sub_cost=0.5f*(grayImageR.at<uchar>(i,j-d)+grayImageR.at<uchar>(i,j-d-1));
                float gray_r_add_cost=0.5f*(grayImageR.at<uchar>(i,j-d)+grayImageR.at<uchar>(i,j-d+1));
                float sobel_l_sub_cost=0.5f*(sobelLimg.at<short>(i,j)+sobelLimg.at<short>(i,j-1));
                float sobel_l_add_cost=0.5f*(sobelLimg.at<short>(i,j)+sobelLimg.at<short>(i,j+1));
                float sobel_r_sub_cost=0.5f*(sobelRimg.at<short>(i,j-d)+sobelRimg.at<short>(i,j-d-1));
                float sobel_r_add_cost=0.5f*(sobelRimg.at<short>(i,j-d)+sobelRimg.at<short>(i,j-d+1));
                float gray_l_min=min(gray_l_add_cost,gray_l_sub_cost);
                gray_l_min=min(gray_l_min,float(grayImageL.at<uchar>(i,j)));
                float gray_l_max=max(gray_l_add_cost,gray_l_sub_cost);
                gray_l_max=max(gray_l_max,float(grayImageL.at<uchar>(i,j)));
                float gray_l_d=max(0.0f,float(grayImageR.at<uchar>(i,j-d))-gray_l_max);
                gray_l_d=max(gray_l_d,gray_l_min-float(grayImageR.at<uchar>(i,j-d)));

                float gray_r_min=min(gray_r_add_cost,gray_r_sub_cost);
                gray_r_min=min(gray_r_min,float(grayImageR.at<uchar>(i,j-d)));
                float gray_r_max=max(gray_r_add_cost,gray_r_sub_cost);
                gray_r_max=max(gray_r_max,float(grayImageR.at<uchar>(i,j-d)));
                float gray_r_d=max(0.0f,float(grayImageL.at<uchar>(i,j))-gray_r_max);
                gray_r_d=max(gray_r_d,gray_r_min-float(grayImageL.at<uchar>(i,j)));
                float gray_cost=gray_l_d+gray_r_d;

                float sobel_l_min=min(sobel_l_add_cost,sobel_l_sub_cost);
                sobel_l_min=min(sobel_l_min,float(sobelLimg.at<short>(i,j)));
                float sobel_l_max=max(sobel_l_add_cost,sobel_l_sub_cost);
                sobel_l_max=max(sobel_l_max,float(sobelLimg.at<short>(i,j)));
                float sobel_l_d=max(0.0f,float(sobelRimg.at<short>(i,j-d))-sobel_l_max);
                sobel_l_d=max(sobel_l_d,sobel_l_min-float(sobelRimg.at<short>(i,j-d)));

                float sobel_r_min=min(sobel_r_add_cost,sobel_r_sub_cost);
                sobel_r_min=min(sobel_r_min,float(sobelRimg.at<short>(i,j-d)));
                float sobel_r_max=max(sobel_r_add_cost,sobel_r_sub_cost);
                sobel_r_max=max(sobel_r_max,float(sobelRimg.at<short>(i,j-d)));
                float sobel_r_d=max(0.0f,float(sobelLimg.at<short>(i,j))-sobel_r_max);
                sobel_r_d=max(sobel_r_d,sobel_r_min-float(sobelLimg.at<short>(i,j)));
                float sobel_cost=sobel_l_d+sobel_r_d;

                cost_array[i][j][d]=gray_cost+sobel_cost;
                //cost_array[i][j][d]=abs(grayImageL.at<uchar>(i,j)-grayImageR.at<uchar>(i,j-d));
            }
        }
    }
}

void compute_disparity()
{
    for(int i=0;i<imageHeight;i++)
    {
        for(int j=0;j<imageWidth;j++)
        {
            int min_cost = 100000;
            int min_d = 0;
            int second_min_cost = 100000;
            int second_min_d = 0;
            int max_cost = 0;
            for(int d=minDisparities;d<=maxDisparities;d++)
            {
                if(cost_array[i][j][d]<min_cost)
                {
                    min_cost = cost_array[i][j][d];
                    min_d = d;
                }
                if(cost_array[i][j][d]>max_cost)
                {
                    max_cost = cost_array[i][j][d];
                }
                if(cost_array[i][j][d]<second_min_cost&&cost_array[i][j][d]>min_cost)
                {
                    second_min_cost = cost_array[i][j][d];
                    second_min_d = d;
                }
            }

        if(uniqueness&&second_min_cost-min_cost<=min_cost*(100-disparity_threshold)/100.0)
        {
            disparity.at<float>(i,j) = 0;
            match_left.at<float>(i,j)=0;
            continue;
        }
        int cost_1 = min_cost;
        int cost_2 = min_cost;
        if(min_d==minDisparities||min_d==maxDisparities)
        {
            disparity.at<float>(i,j) = 0;
            match_left.at<float>(i,j)=0;
            continue;
        }
        else
        {
            cost_1 = cost_array[i][j][min_d-1];
            cost_2 = cost_array[i][j][min_d+1];
        }
        float d = min_d;
        if(sub_pixel){
        d = min_d+0.5f*float(cost_1-cost_2)/max(1.0f,float(cost_1-2*min_cost+cost_2));
        }
        match_left.at<float>(i,j) = d;

        disparity.at<float>(i,j) = d/float(maxDisparities+1);
            
        }
    }
}

void cost_aggregation_left()
{
for(int i=0;i<imageHeight;i++)
    {   int lastmin=100000;
        for(int j=0;j<imageWidth;j++)
        {   int nowmin=100000;
            for(int d=minDisparities;d<=maxDisparities;d++)
            {
                int cost = cost_array[i][j][d];
                if(j==0)
                {
                    left_cost[i][j][d] = cost;
                    if(cost<lastmin)
                        nowmin=cost;
                    continue;
                }
                
                int m1=left_cost[i][j-1][d];
                int m2=100000;
                if(d>0)
                    int m2=left_cost[i][j-1][d-1]+P1;
                int m3=100000;
                if(d<maxDisparities)
                    int m3=left_cost[i][j-1][d+1]+P1;
                int pp2=max(P2/(abs(int(grayImageL.at<uchar>(i,j))-int(grayImageL.at<uchar>(i,j-1)))+1),P1);
                int m4=lastmin+pp2;
                left_cost[i][j][d]=min(min(min(m1,m2),m3),m4)-lastmin+cost;
                if(left_cost[i][j][d]<nowmin)
                    nowmin=left_cost[i][j][d];
            }
            lastmin=nowmin;
        }
        
    }
}

void cost_aggregation_right()
{
for(int i=0;i<imageHeight;i++)
    {   int lastmin=100000;
        for(int j=imageWidth-1;j>=0;j--)
        {
            int nowmin=100000;
            for(int d=minDisparities;d<=maxDisparities;d++)
            {
                int cost = cost_array[i][j][d];
                if(j==imageWidth-1)
                {
                    right_cost[i][j][d] = cost;
                    if(cost<lastmin)
                        nowmin=cost;
                    continue;
                }
                
                int m1=right_cost[i][j+1][d];
                int m2=100000;
                if(d>0)
                    int m2=right_cost[i][j+1][d-1]+P1;
                int m3=100000;
                if(d<maxDisparities)
                    int m3=right_cost[i][j+1][d+1]+P1;
                    int pp2=max(P2/(abs(grayImageR.at<uchar>(i,j)-grayImageR.at<uchar>(i,j+1))+1),P1);
                int m4=lastmin+pp2;
                right_cost[i][j][d]=min(min(min(m1,m2),m3),m4)-lastmin+cost;
                if(right_cost[i][j][d]<nowmin)
                    nowmin=right_cost[i][j][d];
            }
            lastmin=nowmin;
        }
        
    }

}

void cost_aggregation_up()
{
for(int j=0;j<imageWidth;j++)
    {   int lastmin=100000;
        for(int i=0;i<imageHeight;i++)
        {
            int nowmin=100000;
            for(int d=minDisparities;d<=maxDisparities;d++)
            {
                int cost = cost_array[i][j][d];
                if(i==0)
                {
                    up_cost[i][j][d] = cost;
                    if(cost<lastmin)
                        nowmin=cost;
                    continue;
                }
                
                int m1=up_cost[i-1][j][d];
                int m2=100000;
                if(d>0)
                    int m2=up_cost[i-1][j][d-1]+P1;
                int m3=100000;
                if(d<maxDisparities)
                    int m3=up_cost[i-1][j][d+1]+P1;
                    int pp2=P2/(abs(grayImageL.at<uchar>(i,j)-grayImageL.at<uchar>(i-1,j))+1);
                int m4=lastmin+pp2;
                up_cost[i][j][d]=min(min(min(m1,m2),m3),m4)-lastmin+cost;
                if(up_cost[i][j][d]<nowmin)
                    nowmin=up_cost[i][j][d];
            }
            lastmin=nowmin;
        }
        
    }

}

void cost_aggregation_down()
{
for(int j=0;j<imageWidth;j++)
    {   int lastmin=100000;
        for(int i=imageHeight-1;i>=0;i--)
        {
            int nowmin=100000;
            for(int d=minDisparities;d<=maxDisparities;d++)
            {
                int cost = cost_array[i][j][d];
                if(i==imageHeight-1)
                {
                    down_cost[i][j][d] = cost;
                    if(cost<lastmin)
                        nowmin=cost;
                    continue;
                }
                
                int m1=down_cost[i+1][j][d];
                int m2=100000;
                if(d>0)
                    int m2=down_cost[i+1][j][d-1]+P1;
                int m3=100000;
                if(d<maxDisparities)
                    int m3=down_cost[i+1][j][d+1]+P1;
                    int pp2=P2/(abs(grayImageL.at<uchar>(i,j)-grayImageL.at<uchar>(i+1,j))+1);
                int m4=lastmin+pp2;
                down_cost[i][j][d]=min(min(min(m1,m2),m3),m4)-lastmin+cost;
                if(down_cost[i][j][d]<nowmin)
                    nowmin=down_cost[i][j][d];
            }
            lastmin=nowmin;
        }
        
    }

}

void cost_aggregation()
{
    cost_aggregation_left();
    cost_aggregation_right();
    cost_aggregation_up();
    cost_aggregation_down();
for(int i=0;i<imageHeight;i++)
    {
        for(int j=0;j<imageWidth;j++)
        {
            for(int d=minDisparities;d<=maxDisparities;d++)
            {
                cost_array[i][j][d]=left_cost[i][j][d]+right_cost[i][j][d]+up_cost[i][j][d]+down_cost[i][j][d];
            }
        }
    }
}

void compute_rightimg_cost()
{
    for(int i=0;i<imageHeight;i++)
    {
        for(int j=0;j<imageWidth;j++)
        {
            for(int d=minDisparities;d<=maxDisparities;d++)
            {   
                if(j+d>=imageWidth)
                    rightimg_cost[i][j][d]=100000;
                else
                    rightimg_cost[i][j][d]=cost_array[i][j+d][d];
            }
        }
    }
}

void compute_disparity_right()
{   compute_rightimg_cost();
    for(int i=0;i<imageHeight;i++)
    {
        for(int j=0;j<imageWidth;j++)
        {
            int min_cost = 100000;
            int min_d = 0;
            int max_cost = 0;
            int second_min_cost = 100000;
            int second_min_d = 0;
            for(int d=minDisparities;d<=maxDisparities;d++)
            {
                if(rightimg_cost[i][j][d]<min_cost)
                {
                    min_cost = rightimg_cost[i][j][d];
                    min_d = d;
                }
                if(rightimg_cost[i][j][d]>max_cost)
                {
                    max_cost = rightimg_cost[i][j][d];
                }
                if(rightimg_cost[i][j][d]<second_min_cost && rightimg_cost[i][j][d]>min_cost)
                {
                    second_min_cost = rightimg_cost[i][j][d];
                    second_min_d = d;
                }
            }
            if(uniqueness&&second_min_cost-min_cost<min_cost*(100-disparity_threshold)/100.0)
            {
                right_disparity.at<float>(i,j) = 0;
                match_right.at<float>(i,j)=0;
                continue;
            }

            int cost_1 = min_cost;
            int cost_2 = min_cost;
        if(min_d==minDisparities||min_d==maxDisparities)
        {
            right_disparity.at<float>(i,j) = 0;
            match_right.at<float>(i,j)=0;
            continue;
        }
        else
        {
            cost_1 = rightimg_cost[i][j][min_d-1];
            cost_2 = rightimg_cost[i][j][min_d+1];
        }
            float d=min_d;
            if(sub_pixel){
            d= min_d+0.5f*float(cost_1-cost_2)/max(1.0f,float(cost_1-2*min_cost+cost_2));
            }
            match_right.at<float>(i,j) = d;

            right_disparity.at<float>(i,j) = d/float(maxDisparities+1);
                
        }
    }
}

void check_consistence()
{int num=0;
occlusions.clear();
mismatches.clear();
    for(int i=0;i<imageHeight;i++)
    {
        for(int j=0;j<imageWidth;j++)
        {  
            float disp=match_left.at<float>(i,j);
            if(disp==0)
            {
                mismatches.emplace_back(i, j);
                continue;
            }
            float col_right=int(j-disp+0.5);
            
            if(col_right>=0&&col_right<imageWidth)
            {   float disp_r=match_right.at<float>(i,col_right);
                if(abs(disp-disp_r)>consistence_threshold)
                {
                    int col_rl=int(col_right+disp_r+0.5);
                    if(col_rl>0&&col_rl<imageWidth)
                    {
                        float disp_l=match_left.at<float>(i,col_rl);
                        if(disp_l>disp)
                        {
                            occlusions.emplace_back(i, j);
                        }
                        else
                        {
                            mismatches.emplace_back(i, j);
                        }
                    }
                    else
                    {
                        mismatches.emplace_back(i, j);
                    }
                    disparity.at<float>(i,j)=0;
                }
            }
            else
            {
                disparity.at<float>(i,j)=0;
                mismatches.emplace_back(i, j);
            }


        }
    }
    cout<<"mismatch:"<<mismatches.size()<<endl;
    cout<<"occlusion:"<<occlusions.size()<<endl;
}

void remove_little_connected_component()
{   

    for(int i=0;i<imageHeight;i++)
    {
        for(int j=0;j<imageWidth;j++)
        {
            flags[i][j]=false;
        }
    }


    for(int m=0;m<imageHeight;m++)
    {
        for(int n=0;n<imageWidth;n++)
        {
            if(flags[m][n]==true)
            {
                continue;
            }
            int num=0;
            vector<Point> points;
            vector<Point> tobedel;
            points.push_back(Point(n,m));
            flags[m][n]=true;
            while(!points.empty())
            {
                Point p=points.back();
                points.pop_back();
                tobedel.push_back(p);
                int i=p.y;
                int j=p.x;
             for(int ii=-1;ii<2;ii++)
             {
                for(int jj=-1;jj<2;jj++)
                {
                    if(ii==0&&jj==0)
                        continue;
                    if(i+ii<0||i+ii>=imageHeight||j+jj<0||j+jj>=imageWidth)
                        continue;
                    if(flags[i+ii][j+jj]==true)
                        continue;
                    if(abs(disparity.at<float>(i,j)-disparity.at<float>(i+ii,j+jj))>diff_threshold/float(maxDisparities+1))
                    {
                        continue;
                    }
                    points.push_back(Point(j+jj,i+ii));
                    flags[i+ii][j+jj]=true;
                    num++;
                }
             }

            }
            
            if(num<min_size)
            {
                for(int i=0;i<tobedel.size();i++)
                {
                    int ii=tobedel[i].y;
                    int jj=tobedel[i].x;
                    disparity.at<float>(ii,jj)=0;
                }
            }
        }
    }

}

void fill_blanck_area()
{   final_disparity=disparity.clone();
    for(int k=0;k<3;k++)
    {   vector<pair<int,int>> remained;
        vector<pair<int,int>>& to_be_dealed=(k==0)?occlusions:mismatches;
        if(to_be_dealed.empty())
        {
            continue;
        }
        if(k==2)
        {
            for(int i=0;i<imageHeight;i++)
            {
                for(int j=0;j<imageWidth;j++)
                {
                    if(disparity.at<float>(i,j)==0)
                    {
                        remained.emplace_back(i,j);
                    }
                }
            }
            to_be_dealed=remained;
        }



        for(int n=0;n<to_be_dealed.size();n++)
        {   
            
            float around[8]={};
            int i=to_be_dealed[n].first;
            int j=to_be_dealed[n].second;
            for(int d=0;d<maxDisparities&&j-d>=0;d++)
            {
                if(disparity.at<float>(i,j-d)!=0)
                {
                    around[0]=disparity.at<float>(i,j-d);
                    break;
                }
            }
            for(int d=0;d<maxDisparities&&j+d<imageWidth;d++)
            {
                if(disparity.at<float>(i,j+d)!=0)
                {
                    around[1]=disparity.at<float>(i,j+d);
                    break;
                }
            }
            for(int d=0;d<maxDisparities&&i-d>=0;d++)
            {
                if(disparity.at<float>(i-d,j)!=0)
                {
                    around[2]=disparity.at<float>(i-d,j);
                    break;
                }
            }
            for(int d=0;d<maxDisparities&&i+d<imageHeight;d++)
            {
                if(disparity.at<float>(i+d,j)!=0)
                {
                    around[3]=disparity.at<float>(i+d,j);
                    break;
                }
            }
            for(int d=0;d<maxDisparities&&i-d>=0&&j-d>=0;d++)
            {
                if(disparity.at<float>(i-d,j-d)!=0)
                {
                    around[4]=disparity.at<float>(i-d,j-d);
                    break;
                }
            }
            for(int d=0;d<maxDisparities&&i+d<imageHeight&&j+d<imageWidth;d++)
            {
                if(disparity.at<float>(i+d,j+d)!=0)
                {
                    around[5]=disparity.at<float>(i+d,j+d);
                    break;
                }
            }
            for(int d=0;d<maxDisparities&&i-d>=0&&j+d<imageWidth;d++)
            {
                if(disparity.at<float>(i-d,j+d)!=0)
                {
                    around[6]=disparity.at<float>(i-d,j+d);
                    break;
                }
            }
            for(int d=0;d<maxDisparities&&i+d<imageHeight&&j-d>=0;d++)
            {
                if(disparity.at<float>(i+d,j-d)!=0)
                {
                    around[7]=disparity.at<float>(i+d,j-d);
                    break;
                }
            }
            if(k==0)
            {
                float second_min=10000;
                float min=10000;
                for(int m=0;m<8;m++)
                {

                        if(around[m]<min)
                        {
                            second_min=min;
                            min=around[m];
                        }
                        else if(around[m]<second_min)
                        {
                            second_min=around[m];
                        }

                }
                if(second_min!=10000)
                {
                    final_disparity.at<float>(i,j)=second_min;
                }
            }
            else
            {
                float sum=0;
                int num=0;
                for(int m=0;m<8;m++)
                {
                    if(around[m]!=0)
                    {
                        sum+=around[m];
                        num++;
                    }
                }
                if(num!=0)
                {
                    final_disparity.at<float>(i,j)=sum/num;
                }
                // float median=0;
                // vector<float> around_vec;
                // for(int m=0;m<8;m++)
                // {
                //     if(around[m]!=0)
                //     {
                //         around_vec.push_back(around[m]);
                //     }
                // }
                // sort(around_vec.begin(),around_vec.end());
                // if(around_vec.size()%2==0)
                // {
                //     median=(around_vec[around_vec.size()/2]+around_vec[around_vec.size()/2-1])/2;
                // }
                // else
                // {
                //     median=around_vec[around_vec.size()/2];
                // }
                // final_disparity.at<float>(i,j)=median;
            }
        }













    }
}

void my_sgm(int,void*)
{
    compute_cost();
    cost_aggregation();
    compute_disparity();
    if(adjust){
    compute_disparity_right();
    if(is_check_consistence)
    {
    check_consistence();
    }
    if(del_little_area)
    {
    remove_little_connected_component();
    }
    
    fill_blanck_area();
    medianBlur(final_disparity,final_disparity,3);
    imshow("disparity",final_disparity);
    }
    else
    {
    imshow("disparity",disparity);
    }
}


    int main()
    {
        init();
	namedWindow("disparity");

	//createTrackbar("minDisparities:", "disparity", &minDisparities, 16, my_sgm);
    //createTrackbar("maxDisparities:", "disparity", &maxDisparities, 64, my_sgm);

    createTrackbar("adjust:", "disparity", &adjust, 1, my_sgm);
    //createTrackbar("P1:", "disparity", &P1, 50, my_sgm);
    //createTrackbar("P2:", "disparity", &P2, 500, my_sgm);

    createTrackbar("disparity_threshold:", "disparity", &disparity_threshold, 100, my_sgm);

    //createTrackbar("diff_threshold:", "disparity", &diff_threshold, 30, my_sgm);
    //createTrackbar("min_size:", "disparity", &min_size, 100, my_sgm);
    
    createTrackbar("occlusion_threshold", "disparity", &occlusion_threshold, 100, my_sgm);
    createTrackbar("consistence_threshold","disparity",&consistence_threshold,10,my_sgm);
    

	   
 
	waitKey(0);
	return 0;
    }
 