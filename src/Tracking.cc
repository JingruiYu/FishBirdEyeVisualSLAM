/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
//#include<eigen3/Eigen/Core>
//#include<eigen3/Eigen/Geometry>
#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"
#include "simple_birdseye_odometer.h"

#include<iostream>
#include <unistd.h>
#include<mutex>


using namespace std;

extern bool bTightCouple;
extern bool bLooseCouple;
extern bool bHaveBird;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), IsReInit(false),
    Twc_ptr_(new pcl::visualization::PCLVisualizer("Twc_viewer")),
    Twb_ptr_(new pcl::visualization::PCLVisualizer("Twb_viewer")),
    mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    int v1(0);
    Twc_ptr_->createViewPort(0.0, 0.0, 1.0, 1.0, v1);
    Twc_ptr_->createViewPortCamera(v1);
    Twc_ptr_->setBackgroundColor(0, 0, 0, v1);
    Twc_ptr_->addCoordinateSystem(1.0, 0.0, 0.0, 0.3, "vehicle_frame", v1);
    Twc_ptr_->addCoordinateSystem(1.0, 0.0, 0.0, 0.0, "map_frame", v1);

    Twb_ptr_->createViewPort(0.0, 0.0, 1.0, 1.0, v1);
    Twb_ptr_->createViewPortCamera(v1);
    Twb_ptr_->setBackgroundColor(0, 0, 0, v1);
    Twb_ptr_->addCoordinateSystem(1.0, 0.0, 0.0, 0.3, "vehicle_frame", v1);
    Twb_ptr_->addCoordinateSystem(1.0, 0.0, 0.0, 0.0, "map_frame", v1);

    outlierCnt = 0;
    inlierCnt = 0;
    lessMatch = 0;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

/********************* Modified Here *********************/
cv::Mat Tracking::GrabImageMonocularWithOdom(const cv::Mat &im, const cv::Mat &birdview, const cv::Mat &birdviewmask, const cv::Mat &birdviewContour,
                                             const cv::Mat &birdviewContourICP, const double &timestamp, cv::Vec3d odomPose, cv::Vec3d gtPose)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    cv::Mat BirdGray = birdview.clone();

    //Convert bird view to grayscale
    if(BirdGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(BirdGray,BirdGray,CV_RGB2GRAY);
        else
            cvtColor(BirdGray,BirdGray,CV_BGR2GRAY);
    }
    else if(BirdGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(BirdGray,BirdGray,CV_RGBA2GRAY);
        else
            cvtColor(BirdGray,BirdGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,BirdGray,birdviewmask,birdviewContour,birdviewContourICP,timestamp,odomPose,gtPose,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,BirdGray,birdviewmask,birdviewContour,birdviewContourICP,timestamp,odomPose,gtPose,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();
    // TrackB();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::TrackB()
{
    if (!tmpRefFrame)
    {
        tmpTwb = cv::Mat::eye(4,4,CV_32F);
        tmpTwc = cv::Mat::eye(4,4,CV_32F);
        
        mCurrentFrame.testTbw = Converter::invT(tmpTwb.clone());
        mCurrentFrame.SetPose(Converter::invT(tmpTwc));
        
        mCurrentFrame.mvBirdOutlier = vector<bool>(mCurrentFrame.Nbird,false);

        tmpRefFrame = new Frame(mCurrentFrame);

        tmpvFrame.push_back(tmpRefFrame);
        localFrame.push_front(tmpRefFrame);

        GenerateBirdPoints();
        UpdateBirdLocalMap();

        mpFrameDrawer->Update(this);

    }
    else
    {        
        cv::Mat T_from_bi2_to_bi1 = Converter::GetTbi2bi1FromOdometer(tmpRefFrame->mGtPose,mCurrentFrame.mGtPose); 
        tmpTwb = tmpTwb * T_from_bi2_to_bi1;  
        mCurrentFrame.testTbw = Converter::invT(tmpTwb);

        cv::Mat T_from_ci2_to_ci1 = Frame::GetTransformFromOdometer(tmpRefFrame->mGtPose,mCurrentFrame.mGtPose);       
        tmpTwc = tmpTwc * T_from_ci2_to_ci1;

        cv::Mat T_from_ci1_to_ci2 = Converter::invT(T_from_ci2_to_ci1);
        mCurrentFrame.SetPose(T_from_ci1_to_ci2*tmpRefFrame->mTcw);

        int numPt = mCurrentFrame.GetBirdMapPointsNum();
        cout << "the num of MPs -1-: " << numPt << endl;

        ORBmatcher BirdMatcher(0.9,true);
        vector<cv::DMatch> vDMatches12;
        int nmatches;
        if (false)
        {
            nmatches = BirdMatcher.BirdviewMatch(mCurrentFrame,tmpRefFrame->mvKeysBird,tmpRefFrame->mDescriptorsBird,tmpRefFrame->mvpMapPointsBird,vDMatches12,0,10);
            // cout << "nmatches:- " << nmatches << " - vDMatches12.size() : - " << vDMatches12.size() << endl;
            FilterBirdOutlier(tmpRefFrame, &mCurrentFrame, vDMatches12, 0.05);

            numPt = mCurrentFrame.GetBirdMapPointsNum();
            cout << "the num of MPs -2-: " << numPt << endl;
        }
        else
        {
            if (localMapPointBirds.size() > 10)
                nmatches = BirdMatcher.BirdMapPointMatch(mCurrentFrame, localMapPointBirds, 10, 0.05);
            else
                nmatches = BirdMatcher.BirdMapPointMatch(mCurrentFrame, tmpRefFrame->mvpMapPointsBird, 10, 0.05);
            
            numPt = mCurrentFrame.GetBirdMapPointsNum();
            cout << "the num of MPs -2-: " << numPt << endl;

            if (nmatches < 20 || numPt < 20)
            {
                lessMatch++;
            }
            // if (nmatches < 50 || numPt < 50)
            // {
                nmatches = BirdMatcher.BirdviewMatch(mCurrentFrame,tmpRefFrame->mvKeysBird,tmpRefFrame->mDescriptorsBird,tmpRefFrame->mvpMapPointsBird,vDMatches12,0,10);
                FilterBirdOutlier(tmpRefFrame, &mCurrentFrame, vDMatches12, 0.05);
            // }
            // else
            // {
            //     cout << "\033[31m" << "no New !!!" << "\033[0m" << endl;
            //     getchar();
            // }
        
            numPt = mCurrentFrame.GetBirdMapPointsNum();
            cout << "the num of MPs -3-: " << numPt << endl;
        }
        
        
        if (numPt > 10)
        {
            cout << "Tcw before optimization: " << endl << mCurrentFrame.mTcw << endl;

            // CheckOptim(&mCurrentFrame);

            // numPt = mCurrentFrame.GetBirdMapPointsNum();
            // cout << "the num of MPs -4-: " << numPt << endl;

            int optedMatches = Optimizer::BirdOptimization(&mCurrentFrame,1.0);
            cout << "After optimization, matches : " << optedMatches << endl;

            cout << "Tcw after optimization: " << endl << mCurrentFrame.mTcw << endl;

            inlierCnt++;

            numPt = mCurrentFrame.GetBirdMapPointsNum();
            cout << "the num of MPs -4-: " << numPt << endl;

        }
        else
        {
            cout << "\033[32m" << "not enough matches for optimization" << "\033[0m" << endl;
            // mCurrentFrame.SetPose(Converter::invT(tmpTwc));

            outlierCnt++;
        }
        
        nmatches = BirdMatcher.BirdMapPointMatch(mCurrentFrame, localMapPointBirds, 10, 0.05);
        nmatches = BirdMatcher.BirdviewMatch(mCurrentFrame,tmpRefFrame->mvKeysBird,tmpRefFrame->mDescriptorsBird,tmpRefFrame->mvpMapPointsBird,vDMatches12,0,10);
        FilterBirdOutlier(tmpRefFrame, &mCurrentFrame, vDMatches12, 0.05);

        numPt = mCurrentFrame.GetBirdMapPointsNum();
        cout << "the num of MPs -5-: " << numPt << endl;

        vBirdDMatchs.assign(vDMatches12.begin(),vDMatches12.end());
        cout << "vDMatches12:- " << vDMatches12.size() << " - vBirdDMatchs.size() : - " << vBirdDMatchs.size() << endl;

        mpFrameDrawer->Update(this);

        tmpRefFrame = new Frame(mCurrentFrame);
        tmpvFrame.push_back(tmpRefFrame);   
        localFrame.push_front(tmpRefFrame);

        UpdateBirdLocalMap();
    }

    cv::Mat Twb_c2 = Frame::Tbc * tmpTwc;
    cv::Mat Twb_c = Frame::Tbc * Converter::invT(mCurrentFrame.mTcw);
    cout << "Twb_c2 : " << endl << Twb_c2 << endl;
    cout << "Twb_c : " << endl << Twb_c << endl;
    DrawTwb_cPose(Twb_c,0,150,0,"GTwb_c");
    DrawTwbPose(tmpTwb,0,150,0,"GTwb");
    DrawGT(0,0,150,"GroundTruth");

    cout << "inlierCnt: " << inlierCnt << " outlierCnt: " << outlierCnt << " lessMatch: " << lessMatch << endl;
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        tmpRefFrame = new Frame(mInitialFrame);
        mpFrameDrawer->Update(this);
        tmpRefFrame = new Frame(mCurrentFrame);

        if (mState == OK)
        {
            cout << "Initial Frame is : " << mCurrentFrame.mnId << endl;
            std::vector<MapPointBird*> vMPB = mpMap->GetAllMapPointsBird();
            cout << "vMPB in MP is " << vMPB.size() << endl;
            getchar();
        }
        
        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                cv::Mat detlaT = cv::Mat::eye(4,4,CV_32F);
                if(mCurrentFrame.mbHaveOdom && bLooseCouple)
                    detlaT=Frame::GetTransformFromOdometer(mLastFrame.mGtPose,mCurrentFrame.mGtPose).inv();
                else if (!mVelocity.empty())
                    detlaT=mVelocity;
                
                mCurrentFrame.SetPose(detlaT*mLastFrame.mTcw);

                GetPerFrameMatchedBirdPoints();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                // bOK = Relocalization();
                bOK = ReInitialization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        tmpRefFrame = new Frame(mCurrentFrame);
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        DrawCurPose(mCurrentFrame.mTcw,0,150,0,"PoseAfterLocalMap");
        DrawGT(0,0,150,"GroundTruth");

        if(bOK)
            mState = OK;
        else
        {
            mState=LOST;
            IsReInit=false;
        }

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    std::vector<MapPointBird*> vMPB = mpMap->GetAllMapPointsBird();
    cout << "vMPB in MP is " << vMPB.size() << endl;

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;
            // number of RANSAC iterations here
            mpInitializer =  new Initializer(mCurrentFrame,1.0,400);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    cout << "pKFini" << pKFini->mnId << " frameId: " << pKFini->mnFrameId << endl;
    cout << "pKFcur" << pKFcur->mnId << " frameId: " << pKFcur->mnFrameId << endl;

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;
    
    ORBmatcher BirdMatcher(0.9,true);
    vector<cv::DMatch> vDMatches12;
    int nmatches;
    nmatches = BirdMatcher.BirdviewMatch(mCurrentFrame,mInitialFrame.mvKeysBird,mInitialFrame.mDescriptorsBird,mInitialFrame.mvpMapPointsBird,vDMatches12,0,10);
    FilterBirdOutlierInFront(&mInitialFrame, &mCurrentFrame, vDMatches12, 0.05);
    // cout << "the num of MPs -5-: " << mCurrentFrame.GetBirdMapPointsNum() << endl;

    for (size_t i = 0; i < mInitialFrame.mvpMapPointsBird.size(); i++)
    {
        MapPointBird* pMPB = mInitialFrame.mvpMapPointsBird[i];
        if (pMPB)
        {
            pKFini->AddMapPointBird(pMPB,i);
            pMPB->AddObservation(pKFini,i);
            pMPB->ComputeDistinctiveDescriptors();
            
            mpMap->AddMapPointBird(pMPB);
        }
    }

    for (size_t i = 0; i < mCurrentFrame.mvpMapPointsBird.size(); i++)
    {
        MapPointBird* pMPB = mCurrentFrame.mvpMapPointsBird[i];
        if (pMPB)
        {
            pKFcur->AddMapPointBird(pMPB,i);
            pMPB->AddObservation(pKFcur,i);
            pMPB->ComputeDistinctiveDescriptors();

            mpMap->AddMapPointBird(pMPB);
        }
        
    }
    
    
    /********************* Modified Here *********************/
    if(mCurrentFrame.mbHaveOdom&&bTightCouple)
    {
        Optimizer::GlobalBundleAdjustemntWithOdom(mpMap,20);
        // Optimizer::GlobalBundleAdjustemnt(mpMap,20);
    }
    else
    {
        Optimizer::GlobalBundleAdjustemnt(mpMap,20);
    }
    if(!mCurrentFrame.mbHaveOdom)
    {
        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f/medianDepth;

        if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
        {
            cout << "Wrong initialization, reseting..." << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
        for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
        {
            if(vpAllMapPoints[iMP])
            {
                MapPoint* pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}


bool Tracking::CreateReInitialMapPoints()
{
    // Create KeyFrames
    KeyFrame* pKFReI = new KeyFrame(mReInitFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    cout << "pKFini->mnFrameId : " << pKFReI->mnFrameId << endl;
    cout << "pKFcur->mnFrameId : " << pKFcur->mnFrameId << endl;

    pKFReI->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFReI);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFReI->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFReI,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }
    cout << "pKFini->InlierNum : " << pKFReI->GetMapPointsInlierNum() << endl;
    cout << "pKFcur->InlierNum : " << pKFcur->GetMapPointsInlierNum() << endl;

    // Update Connections
    pKFReI->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    // if(mbHaveBirdview)
    // {
    //     Optimizer::GlobalBundleAdjustemntWithBirdview(mpMap,20);
    //     // Optimizer::GlobalBundleAdjustemnt(mpMap,20);
    // }
    // else
    // {
        // Optimizer::GlobalBundleAdjustemnt(mpMap,20);
    // }
    
    cout << "After GlobalBundleAdjustemnt" << endl;
    cout << "pKFini->InlierNum : " << pKFReI->GetMapPointsInlierNum() << endl;
    cout << "pKFcur->InlierNum : " << pKFcur->GetMapPointsInlierNum() << endl;

    // Set median depth to 1
    float medianDepth = pKFReI->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<90)
    {
        cout << "Wrong initialization, reseting..." << endl;
        cout<<"medianDepth = "<<medianDepth<<" , TrackedMapPoint = "<<pKFcur->TrackedMapPoints(1)<<endl;
        
        return false;
    }

    mpLocalMapper->InsertKeyFrame(pKFReI);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFReI);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();

    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFReI);

    mState=OK;
    IsReInit=true;
    return true;
}


void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    /********************* Modified Here *********************/
    // if(mCurrentFrame.mbHaveOdom&&bLooseCouple)
    // {
    //     cv::Mat Tcl=Frame::GetTransformFromOdometer(mLastFrame.mGtPose,mCurrentFrame.mGtPose).inv();
    //     mCurrentFrame.SetPose(Tcl*mLastFrame.mTcw);
    // }
    // else
    // {
    //     mCurrentFrame.SetPose(mLastFrame.mTcw);
    // }
    if (bHaveBird)
        Optimizer::PoseOptimizationWithBird(&mCurrentFrame);
    else
        Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    /********************* Modified Here *********************/
    // if(mLastFrame.mbHaveOdom&&bLooseCouple)
    // {
    //     cv::Mat Tcl=Frame::GetTransformFromOdometer(mLastFrame.mGtPose,mCurrentFrame.mGtPose).inv();
    //     mCurrentFrame.SetPose(Tcl*mLastFrame.mTcw);
    // }
    // else
    // {
    //     mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
    // }

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    if (bHaveBird)
        Optimizer::PoseOptimizationWithBird(&mCurrentFrame);
    else
        Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    GetLocalMapForBird();

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    //int pointsCount=0,inliersCount=0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            //pointsCount++;
            if(!mCurrentFrame.mvbOutlier[i])
            {
                //inliersCount++;
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }
    //cout<<"points: "<<pointsCount<<" , inliers: "<<inliersCount<<" , MatchesInliers: "<<mnMatchesInliers<<endl;

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2 || IsReInit)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    if (IsReInit)
    {
        mpLocalMapper->SetAcceptKeyFrames(true);
    }

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;
    // if(mSensor==System::MONOCULAR)
    //     thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);
    // cout<<"nRefMatches = "<<nRefMatches<<endl;
    // cout<<"thRefRatio = "<<thRefRatio<<endl;
    // cout<<"nRefMatches*thRefRatio = "<<nRefMatches*thRefRatio<<endl;
    // cout<<"mnMatchesInliers = "<<mnMatchesInliers<<endl;
    // cout<<"c1a = "<<c1a<<" , c1b = "<<c1b<<" , c1c = "<<c1c<<" , c2 = "<<c2<<endl;
    // cout<<"(c1a||c1b||c1c)&&c2 = "<<((c1a||c1b||c1c)&&c2)<<endl;
    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            IsReInit = false;
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::GenerateBirdPoints()
{
    cv::Mat Twb = Converter::invT(mCurrentFrame.testTbw);
    cv::Mat Twc = Converter::invT(mCurrentFrame.mTcw);
    for (size_t i = 0; i < mCurrentFrame.mvBirdOutlier.size(); i++)
    {
        MapPointBird * pMP = mCurrentFrame.mvpMapPointsBird[i];
        if (pMP && mCurrentFrame.mvBirdOutlier[i])
        {
            mCurrentFrame.mvpMapPointsBird[i] = static_cast<MapPointBird*>(NULL);
        }
        
        if (!pMP && !mCurrentFrame.mvBirdOutlier[i])
        {
            // cv::Mat pb(mCurrentFrame.mvKeysBirdBaseXY[i]);
            // cv::Mat pw = Twb.rowRange(0,3).colRange(0,3) * pb + Twb.rowRange(0,3).col(3);
            cv::Mat pc(mCurrentFrame.mvKeysBirdCamXYZ[i]);
            cv::Mat pw = Twc.rowRange(0,3).colRange(0,3) * pc + Twc.rowRange(0,3).col(3);
            MapPointBird *pMPBird = new MapPointBird(pw,&mCurrentFrame,mpMap,i);
            mCurrentFrame.mvpMapPointsBird[i] = pMPBird;
        }
    }

    int buildMp = 0;
    for (size_t i = 0; i < mCurrentFrame.mvpMapPointsBird.size(); i++)
    {
        MapPointBird * pMP = mCurrentFrame.mvpMapPointsBird[i];
        if (pMP)
            buildMp++;            
    }
    
    cout << "buildMp: " << buildMp << endl;

    if (buildMp < 10)
    {
        cout << " not right " << endl;
        getchar();
    }
    
}

void Tracking::CreateBirdPoints(KeyFrame* pKF)
{
    int createBMPs = 0;
    cv::Mat Twc = Converter::invT(mCurrentFrame.mTcw);
    for (size_t i = 0; i < mCurrentFrame.mvBirdOutlier.size(); i++)
    {
        MapPointBird * pMP = mCurrentFrame.mvpMapPointsBird[i];
        
        if (!pMP && !mCurrentFrame.mvBirdOutlier[i])
        {
            cv::Mat pc(mCurrentFrame.mvKeysBirdCamXYZ[i]);
            cv::Mat pw = Twc.rowRange(0,3).colRange(0,3) * pc + Twc.rowRange(0,3).col(3);
            MapPointBird *pMPBird = new MapPointBird(pw,pKF,mpMap);
            mCurrentFrame.mvpMapPointsBird[i] = pMPBird;

            createBMPs++;
        }
    }
    cout << "createBMPs: " << createBMPs << endl;
}

void Tracking::FilterBirdOutlier(Frame* MatchedFrame1, Frame* MatchedFrame2, vector<cv::DMatch> &vDMatches12, float windowSize)
{
    vector<cv::KeyPoint> vBirdKey1 = MatchedFrame1->mvKeysBird;
    vector<cv::Point3f> vBirdBaseXYPt1 = MatchedFrame1->mvKeysBirdBaseXY;
    vector<cv::Point3f> vBirdCamXYPt1 = MatchedFrame1->mvKeysBirdCamXYZ;
    cv::Mat Tbw1 = MatchedFrame1->testTbw;
    cv::Mat Twb1 = Converter::invT(Tbw1);
    cv::Mat Tcw1 = MatchedFrame1->mTcw; 
    cv::Mat Twc1 = Converter::invT(Tcw1); 
    cv::Mat Tcw1b = Frame::Tcb * Tbw1 * Frame::Tbc;
    cv::Mat Twc1b = Frame::Tcb * Twb1 * Frame::Tbc;

    vector<cv::KeyPoint> vBirdKey2 = MatchedFrame2->mvKeysBird;
    vector<cv::Point3f> vBirdBaseXYPt2 = MatchedFrame2->mvKeysBirdBaseXY;
    vector<cv::Point3f> vBirdCamXYPt2 = MatchedFrame2->mvKeysBirdCamXYZ;
    cv::Mat Tbw2 = MatchedFrame2->testTbw;
    cv::Mat Twb2 = Converter::invT(Tbw2);
    cv::Mat Tcw2 = MatchedFrame2->mTcw;
    cv::Mat Twc2 = Converter::invT(Tcw2);
    cv::Mat Tcw2b = Frame::Tcb * Tbw2 * Frame::Tbc; 
    cv::Mat Twc2b = Frame::Tcb * Twb2 * Frame::Tbc;
    
    int inlier = 0;
    int buildMP = 0;
    int inConsistent = 0;

    double minDis = INT_MAX;
    double maxDis = INT_MIN;

    double minDisC = INT_MAX;
    double maxDisC = INT_MIN;

    double minDisBC = INT_MAX;
    double maxDisBC = INT_MIN;
    

    for (size_t i = 0; i < vDMatches12.size(); i++)
    {        
        cv::DMatch iMatch = vDMatches12[i];

        MapPointBird * tmpMP = MatchedFrame2->mvpMapPointsBird[iMatch.trainIdx];
        if(tmpMP)
            continue;
            
        cv::Mat pt1(vBirdBaseXYPt1[iMatch.queryIdx]);
        cv::Mat pt2(vBirdBaseXYPt2[iMatch.trainIdx]);

        cv::Mat ptw = Twb1.rowRange(0,3).colRange(0,3) * pt1 + Twb1.rowRange(0,3).col(3);
        cv::Mat ptc2 = Tbw2.rowRange(0,3).colRange(0,3) * ptw + Tbw2.rowRange(0,3).col(3);

        double dis = cv::norm(ptc2-pt2,NORM_L2);


        cv::Mat pt1c(vBirdCamXYPt1[iMatch.queryIdx]);
        cv::Mat pt2c(vBirdCamXYPt2[iMatch.trainIdx]);

        cv::Mat ptwC = Twc1.rowRange(0,3).colRange(0,3) * pt1c + Twc1.rowRange(0,3).col(3);
        cv::Mat ptc2c = Tcw2.rowRange(0,3).colRange(0,3) * ptwC + Tcw2.rowRange(0,3).col(3);

        double disC = cv::norm(ptc2c-pt2c,NORM_L2);


        cv::Mat ptwCb = Twc1b.rowRange(0,3).colRange(0,3) * pt1c + Twc1b.rowRange(0,3).col(3);
        cv::Mat ptc2cb = Tcw2b.rowRange(0,3).colRange(0,3) * ptwCb + Tcw2b.rowRange(0,3).col(3);

        double disCb = cv::norm(ptc2cb-pt2c,NORM_L2);
        

        if (dis < minDis)
            minDis = dis;
        if (dis > maxDis)
            maxDis = dis;

        if (disC < minDisC)
            minDisC = disC;
        if (disC > maxDisC)
            maxDisC = disC;

        if (disCb < minDisBC)
            minDisBC = disCb;
        if (disCb > maxDisBC)
            maxDisBC = disCb;
        
        if (disC < windowSize)
        {
            MatchedFrame2->mvBirdOutlier[iMatch.trainIdx] = false;
            inlier++;            

            MapPointBird *refMPBird = MatchedFrame1->mvpMapPointsBird[iMatch.queryIdx];
            if (refMPBird && cv::norm(refMPBird->GetWorldPos()-ptwC,NORM_L2) < 0.05)
            {
                MatchedFrame2->mvpMapPointsBird[iMatch.trainIdx] = refMPBird;
                buildMP++;
                inConsistent++;
            }
            else
            {
                MapPointBird *pMPBird = new MapPointBird(ptwC,MatchedFrame2,mpMap,iMatch.trainIdx);
                MatchedFrame2->mvpMapPointsBird[iMatch.trainIdx] = pMPBird;
                
                buildMP++;
            }
        }
    }

    cout << "minDis: " << minDis << endl;
    cout << "maxDis: " << maxDis << endl;
    cout << "minDisC: " << minDisC << endl;
    cout << "maxDisC: " << maxDisC << endl;
    cout << "minDisBC: " << minDisBC << endl;
    cout << "maxDisBC: " << maxDisBC << endl;

    cout << "vDMatches12.size(): " << vDMatches12.size() << endl;
    cout << "inlier: " << inlier << endl;
    cout << "buildMP: " << buildMP << endl;
    cout << "inConsistent: " << inConsistent << endl;
    

    // if (minDisC > 1 || minDis > 1 || minDisBC > 1 )
    // {
    //     cout << "Tbw1: " << endl << Tbw1 << endl;
    //     cout << "Tcw1: " << endl << Tcw1 << endl;
    //     cout << "Tcw1b: " << endl << Tcw1b << endl;

    //     cout << "Twb1: " << endl << Twb1 << endl;
    //     cout << "Twc1: " << endl << Twc1 << endl;
    //     cout << "Twc1b: " << endl << Twc1b << endl;
        

    //     cout << "Tbw2: " << endl << Tbw2 << endl;
    //     cout << "Tcw2: " << endl << Tcw2 << endl;
    //     cout << "Tcw2b: " << endl << Tcw2b << endl;

    //     cout << "Twb2: " << endl << Twb2 << endl;
    //     cout << "Twc2: " << endl << Twc2 << endl;
    //     cout << "Twc2b: " << endl << Twc2b << endl;
        
    //     getchar();
    // }
    

}


void Tracking::FilterBirdOutlierInFront(Frame* MatchedFrame1, Frame* MatchedFrame2, vector<cv::DMatch> &vDMatches12, float windowSize)
{
    vector<cv::KeyPoint> vBirdKey1 = MatchedFrame1->mvKeysBird;
    vector<cv::Point3f> vBirdBaseXYPt1 = MatchedFrame1->mvKeysBirdBaseXY;
    vector<cv::Point3f> vBirdCamXYPt1 = MatchedFrame1->mvKeysBirdCamXYZ;
    cv::Mat Tcw1 = MatchedFrame1->mTcw; 
    cout << "Tcw1: " << endl << Tcw1 << endl;
    cv::Mat Twc1 = Converter::invT(Tcw1); 
    cv::Mat Twb1 = MatchedFrame1->GetGTPoseTwb();
    cv::Mat Twc1_fromTwb = Frame::Tcb * Twb1 * Frame::Tbc;

    vector<cv::KeyPoint> vBirdKey2 = MatchedFrame2->mvKeysBird;
    vector<cv::Point3f> vBirdBaseXYPt2 = MatchedFrame2->mvKeysBirdBaseXY;
    vector<cv::Point3f> vBirdCamXYPt2 = MatchedFrame2->mvKeysBirdCamXYZ;
    cv::Mat Tcw2 = MatchedFrame2->mTcw;
    cout << "Tcw2: " << endl << Tcw2 << endl;
    cv::Mat Twc2 = Converter::invT(Tcw2);
    cv::Mat Tbw2 = Converter::invT(MatchedFrame2->GetGTPoseTwb());
    cv::Mat Tcw2_fromTbw = Frame::Tcb * Tbw2 * Frame::Tbc;
    
    int inlier = 0;
    int inConsistent = 0;
    int buildNew = 0;
    int inLierBC = 0;
    double minDisC = INT_MAX;
    double maxDisC = INT_MIN;   
    double minDisCB = INT_MAX;
    double maxDisCB = INT_MIN;  

    vector<cv::DMatch> newMatch;
    for (size_t i = 0; i < vDMatches12.size(); i++)
    {        
        cv::DMatch iMatch = vDMatches12[i];

        MapPointBird * tmpMP = MatchedFrame2->mvpMapPointsBird[iMatch.trainIdx];
        if(tmpMP)
            continue;
            
        cv::Mat pt1c(vBirdCamXYPt1[iMatch.queryIdx]);
        cv::Mat pt2c(vBirdCamXYPt2[iMatch.trainIdx]);

        cv::Mat ptwC = Twc1.rowRange(0,3).colRange(0,3) * pt1c + Twc1.rowRange(0,3).col(3);
        cv::Mat ptc2c = Tcw2.rowRange(0,3).colRange(0,3) * ptwC + Tcw2.rowRange(0,3).col(3);

        cv::Mat ptwCB = Twc1_fromTwb.rowRange(0,3).colRange(0,3) * pt1c + Twc1_fromTwb.rowRange(0,3).col(3);
        cv::Mat ptcb2cb = Tcw2_fromTbw.rowRange(0,3).colRange(0,3) * ptwCB + Tcw2_fromTbw.rowRange(0,3).col(3);

        double disC = cv::norm(ptc2c-pt2c,NORM_L2);
        double disCB = cv::norm(ptcb2cb-pt2c,NORM_L2);

        if (disC < minDisC)
            minDisC = disC;
        if (disC > maxDisC)
            maxDisC = disC;

        if (disCB < minDisCB)
            minDisCB = disCB;
        if (disCB > maxDisCB)
            maxDisCB = disCB;

        if (disC < windowSize)
        {
            MatchedFrame2->mvBirdOutlier[iMatch.trainIdx] = false;
            inlier++;            

            MapPointBird *refMPBird = MatchedFrame1->mvpMapPointsBird[iMatch.queryIdx];
            if (refMPBird && cv::norm(refMPBird->GetWorldPos()-ptwC,NORM_L2) < 0.05)
            {
                MatchedFrame2->mvpMapPointsBird[iMatch.trainIdx] = refMPBird;
                inConsistent++;
            }
            else
            {
                MapPointBird *pMPBird = new MapPointBird(ptwC,MatchedFrame2,mpMap,iMatch.trainIdx);
                MatchedFrame2->mvpMapPointsBird[iMatch.trainIdx] = pMPBird;
                MatchedFrame1->mvpMapPointsBird[iMatch.queryIdx] = pMPBird;
                buildNew++;
            }

            newMatch.push_back(iMatch);
        }

        if (disCB < windowSize)
            inLierBC++;
        
    }

    vBirdDMatchs.assign(newMatch.begin(),newMatch.end());

    cout << "minDisC: " << minDisC << endl;
    cout << "maxDisC: " << maxDisC << endl;
    cout << "minDisCB: " << minDisCB << endl;
    cout << "maxDisCB: " << maxDisCB << endl;

    cout << "vDMatches12.size(): " << vDMatches12.size() << endl;
    cout << "inlier: " << inlier << endl;
    cout << "inLierBC: " << inLierBC << endl;
    cout << "inConsistent: " << inConsistent << endl;
    cout << "buildNew: " << buildNew << endl;
}

void Tracking::CheckOptim(Frame* pFrame)
{
    double error = 0;
    cv::Mat Rcw = pFrame->mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = pFrame->mTcw.rowRange(0,3).col(3);
    
    for (size_t i = 0; i < pFrame->mvpMapPointsBird.size(); i++)
    {
        MapPointBird * pMP = pFrame->mvpMapPointsBird[i];
        if (pMP)
        {
            cv::Mat Xw = pMP->GetWorldPos();
            cv::Mat Xc(pFrame->mvKeysBirdCamXYZ[i]);

            cv::Mat Xrc = Rcw * Xw + tcw;

            double subError = cv::norm(Xrc-Xc,NORM_L2);
            error = subError + error;

            // cout << "Xw: " << Xw.t() << endl;
            // cout << "Xc: " << Xc.t() << endl;
            // cout << "Xrc: " << Xrc.t() << endl;
            // cout << "subError: " << subError << endl;
        } 
    }  
    // cout << "error: " << error << endl;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::GetLocalMapForBird()
{
    map<KeyFrame*,int> keyframeBird;
    int obserMax = INT_MIN;
    int obserNum = 0;
    for (int i = 0; i < mCurrentFrame.Nbird; i++)
    {
        obserNum = 0;
        if (mCurrentFrame.mvpMapPointsBird[i])
        {
            MapPointBird* pMPB = mCurrentFrame.mvpMapPointsBird[i];
            if (!pMPB->isBad())
            {
                const map<KeyFrame*,size_t> observation = pMPB->GetObservations();
                for (map<KeyFrame*,size_t>::const_iterator it=observation.begin(), itend=observation.end(); it!=itend; it++)
                {
                    keyframeBird[it->first]++;
                    obserNum++;
                }
            }
        }
        if (obserNum > obserMax)
            obserMax = obserNum;
    }
    
    cout << "obserMax : " << obserMax << endl;

    if (obserMax > 2)
        getchar();
    
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}


bool Tracking::ReInitialization()
{
    if (mLastFrame.mTcw.empty())
        cout << "mLastFrame.mTcw is empty, which may not exit? " << endl;
    
    mVelocity = GetPriorMotion();
    if (mVelocity.empty())
        cout << "mVelocity is empty, why? " << endl;
    
    // mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
    cv::Mat Twb_now  = mCurrentFrame.GetGTPoseTwb();
    cv::Mat Tc_wc = Converter::Twb2Tcw(Twb_now);
    mCurrentFrame.SetPose(Tc_wc);

    if (!mpReInitial)
    {
        if (mCurrentFrame.mvKeys.size()>100)
        {
            mReInitFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
            
            if (mpReInitial)
                delete mpReInitial;
            
            mpReInitial = new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);           
        }
    }
    else
    {
        if (mCurrentFrame.mvKeys.size()<100)
        {
            delete mpReInitial;
            mpReInitial = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return false;
        }

        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mReInitFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
        
        if (nmatches<100)
        {
            delete mpReInitial;
            mpReInitial = static_cast<Initializer*>(NULL);
            return false;
        }

        cv::Mat R21;
        cv::Mat t21;
        vector<bool> vbTriangulated;
        bool isRe = false;
        if (mpReInitial->ReInitialize(mReInitFrame,mCurrentFrame,mvIniMatches,R21,t21,mvIniP3D,vbTriangulated))
        {
            for (size_t i = 0; i < mvIniMatches.size(); i++)
            {
                if (mvIniMatches[i] >=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }
            
            cv::Mat T21 = cv::Mat::eye(4,4,CV_32F);
            R21.copyTo(T21.rowRange(0,3).colRange(0,3));
            t21.copyTo(T21.rowRange(0,3).col(3));
            cv::Mat Tcw = T21 * mReInitFrame.mTcw;
            mCurrentFrame.SetPose(Tcw);

            isRe = CreateReInitialMapPoints();
        }

        return isRe;
    }  

    return false;  
}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::DrawTwb_cPose(const cv::Mat &Twb_c, double r, double g, double b, string name)
{
    string waypoint_name = name + to_string(mCurrentFrame.mnId);
    
    DrawInTwc_ptr_(Twb_c,r,g,b,waypoint_name);
}

void Tracking::DrawTwbPose(const cv::Mat &Twb, double r, double g, double b, string name)
{
    string waypoint_name = name + to_string(mCurrentFrame.mnId);
    
    DrawInTwb_ptr_(Twb,r,g,b,waypoint_name);
}

void Tracking::DrawCurPose(const cv::Mat &Tcw, double r, double g, double b, string name)
{
    cv::Mat Cur_Twb_c = Converter::Tcw2Twb_c(Tcw);
    string waypoint_name = name + to_string(mCurrentFrame.mnId);
    
    DrawInTwc_ptr_(Cur_Twb_c,r,g,b,waypoint_name);
}

void Tracking::DrawGT(double r, double g, double b, string name)
{
    cv::Mat GT_Twb = mCurrentFrame.GetGTPoseTwb();
    cv::Mat GT_Twb_c = Converter::Twb2Twb_c(GT_Twb);
    string waypoint_name = name + to_string(mCurrentFrame.mnId);
    DrawInTwc_ptr_(GT_Twb_c,r,g,b,waypoint_name);
    DrawInTwb_ptr_(GT_Twb,r,g,b,waypoint_name);
    Twc_ptr_->spinOnce();
    Twb_ptr_->spinOnce();
}


void Tracking::DrawInTwc_ptr_(const cv::Mat &T, double r, double g, double b, string name)
{
    Eigen::Affine3f Draw_pose;
    Draw_pose.matrix() = Converter::toMatrix4f(T);
    birdseye_odometry::SemanticPoint Tpoint;
    Tpoint.x = Draw_pose.translation()[0];
    Tpoint.y = Draw_pose.translation()[1];
    Tpoint.z = Draw_pose.translation()[2];
    Twc_ptr_->addSphere(Tpoint, 0.1, r, g, b, name);
}


void Tracking::DrawInTwb_ptr_(const cv::Mat &T, double r, double g, double b, string name)
{
    Eigen::Affine3f Draw_pose;
    Draw_pose.matrix() = Converter::toMatrix4f(T);
    birdseye_odometry::SemanticPoint Tpoint;
    Tpoint.x = Draw_pose.translation()[0];
    Tpoint.y = Draw_pose.translation()[1];
    Tpoint.z = Draw_pose.translation()[2];
    Twb_ptr_->addSphere(Tpoint, 0.1, r, g, b, name);
}

cv::Mat Tracking::GetPriorMotion()
{
    cv::Mat Twb_last = mLastFrame.GetGTPoseTwb(); // mLastFrame.GetOdomPoseTwb(); //mLastFrame.GetGTPoseTwb();
    cv::Mat Twb_now  = mCurrentFrame.GetGTPoseTwb(); // mCurrentFrame.GetOdomPoseTwb(); //mCurrentFrame.GetGTPoseTwb();
    cv::Mat Twc_last = Converter::Twb2Twc(Twb_last);
    cv::Mat Tcw_now = Converter::Twb2Tcw(Twb_now);

    cv::Mat deltaTcw = Tcw_now*Twc_last;

    // delta cannot preIntergert.
    // cv::Mat Tb2b1 = Converter::invT(Twb_now) * Twb_last;
    // cv::Mat Tc2c1 = Frame::Tcb * Tb2b1 * Frame::Tbc;
    // cout << "Tb2b1 : " << Tb2b1 << endl << "norm is " << norm(Tb2b1.rowRange(0,3).col(3)) << endl;
    // cout << "Tc2c1 : " << Tc2c1 << endl << "norm is " << norm(Tc2c1.rowRange(0,3).col(3)) << endl;
    // cout << "deltaTcw : " << deltaTcw << endl << "norm is " << norm(deltaTcw.rowRange(0,3).col(3)) << endl;

    return deltaTcw.clone();
}

void Tracking::UpdateBirdLocalMap()
{
    cv::Mat nowTw = Converter::invT(mCurrentFrame.mTcw).rowRange(0,3).col(3);
    
    // cout << "localFrame.size() before: " << localFrame.size() << endl;
    // for (auto ite = localFrame.begin(), lend = localFrame.end(); ite != lend; ite++)
    // {
    //     Frame* tmF = *ite;
    //     cout << "id: " << tmF->mnId << endl;
    // }

    int allMapPtBirds = 0;
    set<MapPointBird*> setlocalMapPointBirds;
    for (auto itF = localFrame.begin(), lend = localFrame.end(); itF != lend; )
    {
        Frame* tmF = *itF;
        cv::Mat refTw = Converter::invT(tmF->mTcw).rowRange(0,3).col(3);
        double dis = cv::norm(refTw-nowTw,NORM_L2);
        // cout << "refTw: " << refTw.t() << " nowTw: " << nowTw.t() << " norm: " << dis << endl;
        
        if (cv::norm(refTw-nowTw,NORM_L2) > 5)
        {
            // auto delF = itF;
            // itF++;
            // localFrame.erase(delF);
            localFrame.erase(itF++);
        }
        else
        {
            std::vector<MapPointBird*> submvMPBirds = tmF->mvpMapPointsBird;
            int impMp = 0;
            for (size_t i = 0; i < submvMPBirds.size(); i++)
            {
                MapPointBird* pMPBird = submvMPBirds[i];
                if (!pMPBird)
                    continue;
                
                if (!setlocalMapPointBirds.count(pMPBird))
                {
                    setlocalMapPointBirds.insert(pMPBird);
                    impMp++;
                }
                    
                allMapPtBirds++;
            }
            
            if (impMp < 10)
                localFrame.erase(itF++);
            else
                itF++;
        }        
    }

    cout << "localFrame.size() after : " << localFrame.size() << endl;
    for (auto ite = localFrame.begin(), lend = localFrame.end(); ite != lend; ite++)
    {
        Frame* tmF = *ite;
        cout << "id: " << tmF->mnId << endl;
    }
    
    localMapPointBirds.clear();
    for (auto itMPB = setlocalMapPointBirds.begin(), lend = setlocalMapPointBirds.end(); itMPB != lend; itMPB++)
    {
        MapPointBird* pMPBird = *itMPB;

        if (!pMPBird)
        {
            cout << "pMPBird is NULL" << endl;
            getchar();
        }

        localMapPointBirds.push_back(pMPBird);
    }

    // cout << "allMapPtBirds: " << allMapPtBirds << endl;
    // cout << "setlocalMapPointBirds.size(): " << setlocalMapPointBirds.size() << endl;
    // cout << "localMapPointBirds.size(): " << localMapPointBirds.size() << endl;
}


void Tracking::GetPerFrameMatchedBirdPoints()
{
    ORBmatcher BirdMatcher(0.9,true);
    vector<cv::DMatch> vDMatches12;
    int nmatchesBird = BirdMatcher.BirdviewMatch(mCurrentFrame,tmpRefFrame->mvKeysBird,tmpRefFrame->mDescriptorsBird,tmpRefFrame->mvpMapPointsBird,vDMatches12,0,10);
    FilterBirdOutlierInFront(tmpRefFrame, &mCurrentFrame, vDMatches12, 0.05);
}

} //namespace ORB_SLAM
