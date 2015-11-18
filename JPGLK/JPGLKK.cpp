#include "cv.h" 
#include "highgui.h"

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <assert.h> 
#include <math.h> 
#include <float.h> 
#include <limits.h> 
#include <time.h> 
#include <ctype.h>
/* - 下面这一段，就是预编译指令直接处理了引入库lib，如果是Debug就引用Debug版本，否则引用Release版本 - */
/*#ifdef _DEBUG
#pragma comment(lib,"opencv_ml249d.lib")
#pragma comment(lib,"opencv_calib3d249d.lib")
#pragma comment(lib,"opencv_contrib249d.lib")
#pragma comment(lib,"opencv_core249d.lib")
#pragma comment(lib,"opencv_features2d249d.lib")
#pragma comment(lib,"opencv_flann249d.lib")
#pragma comment(lib,"opencv_gpu249d.lib")
#pragma comment(lib,"opencv_highgui249d.lib")
#pragma comment(lib,"opencv_imgproc249d.lib")
#pragma comment(lib,"opencv_legacy249d.lib")
#pragma comment(lib,"opencv_objdetect249d.lib")
#pragma comment(lib,"opencv_ts249d.lib")
#pragma comment(lib,"opencv_video249d.lib")
#pragma comment(lib,"opencv_nonfree249d.lib")
#pragma comment(lib,"opencv_ocl249d.lib")
#pragma comment(lib,"opencv_photo249d.lib")
#pragma comment(lib,"opencv_stitching249d.lib")
#pragma comment(lib,"opencv_superres249d.lib")
#pragma comment(lib,"opencv_videostab249d.lib")

#else
#pragma comment(lib,"opencv_ml249.lib")
#pragma comment(lib,"opencv_calib3d249.lib")
#pragma comment(lib,"opencv_contrib249.lib")
#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"opencv_features2d249.lib")
#pragma comment(lib,"opencv_flann249.lib")
#pragma comment(lib,"opencv_gpu249.lib")
#pragma comment(lib,"opencv_highgui249.lib")
#pragma comment(lib,"opencv_imgproc249.lib")
#pragma comment(lib,"opencv_legacy249.lib")
#pragma comment(lib,"opencv_objdetect249.lib")
#pragma comment(lib,"opencv_ts249.lib")
#pragma comment(lib,"opencv_video249.lib")
#pragma comment(lib,"opencv_nonfree249.lib")
#pragma comment(lib,"opencv_ocl249.lib")
#pragma comment(lib,"opencv_photo249.lib")
#pragma comment(lib,"opencv_stitching249.lib")
#pragma comment(lib,"opencv_superres249.lib")
#pragma comment(lib,"opencv_videostab249.lib")
#endif
*/
#ifdef _EiC 
#define WIN32 
#endif
static CvMemStorage* storage = 0; 
static CvHaarClassifierCascade* cascade = 0;

void detect_and_draw( IplImage* image );

const char* cascade_name = ""; 

int main( int argc, char** argv ) 
{
    // - 下面这个地址是不能错的，这是你安装的OpenCV库下的一个文件的地址，如果出错，无法识别人脸 
    cascade_name = "D:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";
 cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 ); 
  
    if( !cascade ) 
    { 
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" ); 
        return -1; 
    } 
    storage = cvCreateMemStorage(0); 
    cvNamedWindow( "人脸识别", 1 ); 
     
    const char* filename = "ppc.jpg"; // - 待识别人脸的图片的路径
    IplImage* image = cvLoadImage( filename, 1 );

    if( image ) 
    { 
        detect_and_draw( image ); 
        cvWaitKey(0); 
        cvReleaseImage( &image );   
    }

    cvDestroyWindow("人脸识别"); 
  
    return 0; 
}


void detect_and_draw(IplImage* img ) 
{ 
    double scale=1.2; 
    static CvScalar colors[] = { 
        {{0,0,255}},{{0,128,255}},{{0,255,255}},{{0,255,0}}, 
        {{255,128,0}},{{255,255,0}},{{255,0,0}},{{255,0,255}} 
    };//Just some pretty colors to draw with

    // - 图片预处理
    IplImage* gray = cvCreateImage(cvSize(img->width,img->height),8,1); 
    IplImage* small_img=cvCreateImage(cvSize(cvRound(img->width/scale),cvRound(img->height/scale)),8,1); 
    cvCvtColor(img,gray, CV_BGR2GRAY); 
    cvResize(gray, small_img, CV_INTER_LINEAR);

    cvEqualizeHist(small_img,small_img); //直方图均衡

    //Detect objects if any 
    // 
    cvClearMemStorage(storage); 
    double t = (double)cvGetTickCount(); 
    CvSeq* objects = cvHaarDetectObjects(small_img, 
  cascade,storage,1.1,2,0/*CV_HAAR_DO_CANNY_PRUNING*/,cvSize(30,30));

    t = (double)cvGetTickCount() - t; 
    printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );

    //Loop through found objects and draw boxes around them 
    for(int i=0;i<(objects? objects->total:0);++i) 
    { 
        CvRect* r=(CvRect*)cvGetSeqElem(objects,i); 
        cvRectangle(img, cvPoint(r->x*scale,r->y*scale), cvPoint((r->x+r->width)*scale,(r->y+r->height)*scale), colors[i%8]); 
    } 
    for( int i = 0; i < (objects? objects->total : 0); i++ ) 
    { 
        CvRect* r = (CvRect*)cvGetSeqElem( objects, i ); 
        CvPoint center; 
        int radius; 
        center.x = cvRound((r->x + r->width*0.5)*scale); 
        center.y = cvRound((r->y + r->height*0.5)*scale); 
        radius = cvRound((r->width + r->height)*0.25*scale); 
        cvCircle( img, center, radius, colors[i%8], 3, 8, 0 ); 
    }

    cvShowImage( "人脸识别", img ); 
    cvReleaseImage(&gray); 
    cvReleaseImage(&small_img); 
}