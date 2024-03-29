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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{
std::vector<double> Frame::vTimesFront, Frame::vTimesBird;
long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;
/********************* Modified Here *********************/
cv::Mat Frame::Tbc,Frame::Tcb;
float Frame::mfGridElementWidthInvBirdview, Frame::mfGridElementHeightInvBirdview;
int Frame::birdviewRows, Frame::birdviewCols;

const double correction = 1;
const double Frame::pixel2meter = 0.03984*correction;
const double Frame::meter2pixel = 25.1/correction;
const double Frame::rear_axle_to_center = 1.393;
const double Frame::vehicle_length = 4.63;
const double Frame::vehicle_width = 1.901;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), 
     mvKeysBird(frame.mvKeysBird), mvKeysBirdCamXYZ(frame.mvKeysBirdCamXYZ), mvKeysBirdBaseXY(frame.mvKeysBirdBaseXY), 
     mvpMapPointsBird(frame.mvpMapPointsBird), mDescriptorsBird(frame.mDescriptorsBird), mvBirdOutlier(frame.mvBirdOutlier),
     mEdgeSign(frame.mEdgeSign), mEdgeFree(frame.mEdgeFree),
     mvuRight(frame.mvuRight), mvDepth(frame.mvDepth), 
     mImg(frame.mImg), mBirdColor(frame.mBirdColor), mBirdviewImg(frame.mBirdviewImg), mBirdviewMask(frame.mBirdviewMask),
     mBirdviewContour(frame.mBirdviewContour), mBirdviewContourICP(frame.mBirdviewContourICP),
     mOdomPose(frame.mOdomPose), mGtPose(frame.mGtPose),mbHaveOdom(frame.mbHaveOdom),
     mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), 
     testTbw(frame.testTbw),
     mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    for(int i=0;i<FRAME_GRID_BIRD;i++)
        for(int j=0; j<FRAME_GRID_BIRD; j++)
            mGridBirdview[i][j]=frame.mGridBirdview[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL)),mbHaveOdom(false)
{
    // Frame ID
    mnId=nNextId++;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;
        /********************* Modified Here *********************/
        CalculateExtrinsics();

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),mbHaveOdom(false)
{
    // Frame ID
    mnId=nNextId++;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        /********************* Modified Here *********************/
        CalculateExtrinsics();

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mbHaveOdom(false)
{
    // Frame ID
    mnId=nNextId++;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        /********************* Modified Here *********************/
        CalculateExtrinsics();

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();
}

/********************* Modified Here *********************/
Frame::Frame(const cv::Mat &imGray, const cv::Mat &BirdGray, const cv::Mat &BirdColor, const cv::Mat &birdviewmask, const cv::Mat &birdviewContour,
        const cv::Mat &birdviewContourICP, const double &timeStamp, cv::Vec3d odomPose, cv::Vec3d gtPose,
        ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mOdomPose(odomPose), mGtPose(gtPose), mbHaveOdom(true)
{
    // Frame ID
    mnId=nNextId++;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        birdviewCols = BirdGray.cols;
        birdviewRows = BirdGray.rows;

        mfGridElementWidthInvBirdview=static_cast<float>(FRAME_GRID_BIRD)/static_cast<float>(birdviewCols);
        mfGridElementHeightInvBirdview=static_cast<float>(FRAME_GRID_BIRD)/static_cast<float>(birdviewRows);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        CalculateExtrinsics();

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

// std::chrono:: steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // ORB extraction
    ExtractORB(0,imGray);
// std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
// double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
// vTimesFront.push_back(ttrack);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    mImg = imGray.clone();
    mBirdColor = BirdColor.clone();
    mBirdviewImg = BirdGray.clone();
    mBirdviewMask = birdviewmask.clone();
    mBirdviewContour = birdviewContour.clone();
    mBirdviewContourICP = birdviewContourICP.clone();

// std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    cv::Ptr<cv::ORB> extractorBird = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> preKeysBird;
    extractorBird->detect(mBirdviewImg,preKeysBird,mBirdviewMask);
    // extractorBird->detect(mBirdviewImg,mvKeysBird);

    GuidenceKeyBirdPts(preKeysBird);
    Nbird = mvKeysBird.size();

    vector<cv::Point2f> vKeysBird(Nbird);
    for(int k=0;k<Nbird;k++)
        vKeysBird[k] = mvKeysBird[k].pt;
    
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER,40,0.001);
    cv::cornerSubPix(mBirdviewImg,vKeysBird,cv::Size(5,5),cv::Size(-1,-1),criteria);
    for(int k=0;k<Nbird;k++)
        mvKeysBird[k].pt = vKeysBird[k];


    extractorBird->compute(mBirdviewImg,mvKeysBird,mDescriptorsBird);
// std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
// ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
// vTimesBird.push_back(ttrack);

    mvpMapPointsBird = vector<MapPointBird*>(Nbird,static_cast<MapPointBird*>(NULL));  
    mvBirdOutlier = vector<bool>(Nbird,true);

    mvKeysBirdCamXYZ.resize(Nbird);
    mvKeysBirdBaseXY.resize(Nbird);
    for(int k=0;k<Nbird;k++)
    {
        cv::Point3f p3d = Converter::BirdPixel2BaseXY(mvKeysBird[k]);
        cv::Point2f p2d;
        p2d.x = p3d.x;
        p2d.y = p3d.y;
        mvKeysBirdBaseXY[k] = p3d;
        mvKeysBirdCamXYZ[k] = Converter::BaseXY2CamXYZ(p3d);
    }


    AssignFeaturesToGrid();

    // CalExTime();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }


    int nReserveBird = 0.5f*Nbird/(FRAME_GRID_BIRD*FRAME_GRID_BIRD);
    for(unsigned int i=0; i<FRAME_GRID_BIRD;i++)
        for (unsigned int j=0; j<FRAME_GRID_BIRD;j++)
            mGridBirdview[i][j].reserve(nReserveBird);

    for(int i=0;i<Nbird;i++)
    {
        const cv::KeyPoint &kp = mvKeysBird[i];

        int nGridPosX, nGridPosY;
        if(PosInGridBirdview(kp,nGridPosX,nGridPosY))
            mGridBirdview[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}  

bool Frame::PosInGridBirdview(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round(kp.pt.x*mfGridElementWidthInvBirdview);
    posY = round(kp.pt.y*mfGridElementHeightInvBirdview);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_BIRD || posY<0 || posY>=FRAME_GRID_BIRD)
        return false;

    return true;
} 

vector<size_t> Frame::GetFeaturesInAreaBirdview(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(mvKeysBird.size());

    const int nMinCellX = max(0,(int)floor((x-r)*mfGridElementWidthInvBirdview));
    if (nMinCellX >= FRAME_GRID_BIRD)
        return vIndices;
    
    const int nMaxCellX = min((int)FRAME_GRID_BIRD-1,(int)ceil((x+r)*mfGridElementWidthInvBirdview));
    if (nMaxCellX < 0)
        return vIndices;
    
    const int nMinCellY = max(0,(int)floor((y-r)*mfGridElementHeightInvBirdview));
    if (nMinCellY >= FRAME_GRID_BIRD)
        return vIndices;
    
    const int nMaxCellY = min((int)FRAME_GRID_BIRD-1,(int)ceil((y+r)*mfGridElementHeightInvBirdview));
    if (nMaxCellY < 0)
        return vIndices;
    
    const bool bCheckLevels = (minLevel>0) || (maxLevel >= 0);

    for (int ix = nMinCellX; ix < nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy < nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGridBirdview[ix][iy];
            if (vCell.empty())
                continue;
            
            for (size_t j = 0; j < vCell.size(); j++)
            {
                const cv::KeyPoint &kp = mvKeysBird[vCell[j]];

                if (bCheckLevels)
                {
                    if (kp.octave<minLevel)
                        continue;
                    if (maxLevel>=0)
                        if (kp.octave>maxLevel)
                            continue;                                           
                }

                const float disx = kp.pt.x - x;
                const float disy = kp.pt.y - y;

                if (fabs(disx)<r && fabs(disy)<r)
                    vIndices.push_back(vCell[j]);                
            }
        }
    }

    return vIndices;
}

void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}


void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    //cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    cv::fisheye::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::GuidenceKeyBirdPts(std::vector<cv::KeyPoint>& preKeysBird)
{
    genEdgesPC();

    for (size_t i=0; i<preKeysBird.size(); i++)
    {
        cv::KeyPoint kpt = preKeysBird[i];
        if (nearEdges(kpt))
        {
            mvKeysBird.push_back(kpt);
        }        
        // std::cout << "GuidenceKeyBirdPts: " << mnId << " - " <<  i << std::endl;
    }
}

void Frame::genEdgesPC()
{
    for (size_t row = 0; row < mBirdviewContourICP.rows; row++)
    {
        for (size_t col = 0; col < mBirdviewContourICP.cols; col++)
        {
            int label = -1;

            if (mBirdviewContourICP.at<uchar>(row, col) < 10)
                continue; // free
            else if (mBirdviewContourICP.at<uchar>(row, col) < 150)
                label = 0; // edge
            else
                label = 1; // freespace
            
            cv::Point2f pt;
            pt.x = col;
            pt.y = row;

            if (label)
            {
                mEdgeFree.push_back(pt);
            }
            else
            {
                mEdgeSign.push_back(pt);
            }        
        }
    }  
}

bool Frame::nearEdges(cv::KeyPoint kpt)
{
    int r = 10;
    float pt1x = (kpt.pt.x-r) > 0 ? (kpt.pt.x-r) : 0;
    float pt1y = (kpt.pt.y-r) > 0 ? (kpt.pt.y-r) : 0;
    float pt2x = (kpt.pt.x+r) < mBirdviewContourICP.cols ? (kpt.pt.x+r) : mBirdviewContourICP.cols;
    float pt2y = (kpt.pt.y+r) < mBirdviewContourICP.rows ? (kpt.pt.y+r) : mBirdviewContourICP.rows;
    
    for (size_t row = pt1x; row < pt2x; row++)
    {
        for (size_t col = pt1y; col < pt2y; col++)
        {
            if (mBirdviewContourICP.at<uchar>(row, col) < 10)
                continue; // free
            else if (mBirdviewContourICP.at<uchar>(row, col) < 150)
                return true; // edge
            else
                return true; // freespace
        }
    }

    return false;
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        //cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        cv::fisheye::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        // mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        // mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        // mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

        mnMinX = std::numeric_limits<float>::max();
        mnMaxX = std::numeric_limits<float>::min();
        mnMinY = std::numeric_limits<float>::max();
        mnMaxY = std::numeric_limits<float>::min();

        for (int i = 0; i < 4; ++i)
        {
          if (mat.at<float>(i,0) < mnMinX)
            mnMinX = mat.at<float>(i,0);

          if (mat.at<float>(i,0) > mnMaxX)
            mnMaxX = mat.at<float>(i,0);

          if (mat.at<float>(i,1) < mnMinY)
            mnMinY = mat.at<float>(i,1);

          if (mat.at<float>(i,1) > mnMaxY)
            mnMaxY = mat.at<float>(i,1);
        }
#ifdef DEBUG
        std::cout << "mnMinX: " << mnMinX << std::endl;
        std::cout << "mnMaxX: " << mnMaxX << std::endl;
        std::cout << "mnMinY: " << mnMinY << std::endl;
        std::cout << "mnMaxY: " << mnMaxY << std::endl;
#endif
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0-L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

/********************* Modified Here *********************/

void Frame::CalculateExtrinsics()
{
    // from front camera to base footprint
    cv::Mat tbc = (cv::Mat_<float>(3,1)<<3.747, 0.040, 0.736);
    double qx=0.631,qy=-0.623,qz=0.325,qw=-0.330; 
    double qnorm = sqrt(qx*qx+qy*qy+qz*qz+qw*qw);
    qx = qx / qnorm;
    qy = qy / qnorm;
    qz = qz / qnorm;
    qw = qw / qnorm;   
    cv::Mat Rbc=(cv::Mat_<float>(3,3)<<1-2*(qy*qy+qz*qz),  2*(qx*qy-qw*qz),    2*(qx*qz+qw*qy),
                                                2*(qx*qy+qw*qz),  1-2*(qx*qx+qz*qz),  2*(qy*qz-qw*qx),
                                                2*(qx*qz-qw*qy),  2*(qy*qz+qw*qx),    1-2*(qx*qx+qy*qy));
    Tbc = cv::Mat::eye(4,4,CV_32F);
    Rbc.copyTo(Tbc.rowRange(0,3).colRange(0,3));
    tbc.copyTo(Tbc.rowRange(0,3).col(3));

    Tcb = cv::Mat::eye(4,4,CV_32F);
    cv::Mat Rcb = Rbc.t();
    // cv::Mat Rcb = Rbc.inv();
    cv::Mat tcb = -Rcb*tbc;
    Rcb.copyTo(Tcb.rowRange(0,3).colRange(0,3));
    tcb.copyTo(Tcb.rowRange(0,3).col(3));

    cout<<"extrinsics: "<<endl;
    cout<<"Tbc = "<<endl<<Tbc<<endl;
    cout<<"Tcb = "<<endl<<Tcb<<endl;
    cout << "Is I ? " << endl << Tbc*Tcb << endl;

    // cv::Mat Tcb2 = Tbc.inv();
    // cout<<"Tcb2 = "<<endl<<Tcb2<<endl;
    // cout << "Is I ? " << endl << Tbc*Tcb2 << endl;
}

cv::Mat Frame::GetTransformFromOdometer(const cv::Vec3d &odomPose1, const cv::Vec3d &odomPose2)
{
    //odometer pose
    double x1=odomPose1[0],y1=odomPose1[1],theta1=odomPose1[2];
    double x2=odomPose2[0],y2=odomPose2[1],theta2=odomPose2[2];

    //pre-integration terms
    double theta12=theta2-theta1;
    double x12=(x2-x1)*cos(theta1)+(y2-y1)*sin(theta1);
    double y12=(y2-y1)*cos(theta1)-(x2-x1)*sin(theta1);

    //T12
    cv::Mat T12b=(cv::Mat_<float>(4,4)<<cos(theta12),-sin(theta12),0,x12,
                                        sin(theta12), cos(theta12),0,y12,
                                             0,            0,      1, 0,
                                             0,            0,      0, 1);
    cv::Mat T12c=Tcb*T12b*Tbc;
    return T12c;
}

cv::Mat Frame::GetGTPoseTwb()
{
    double x = mGtPose[0], y = mGtPose[1], theta = mGtPose[2];

    cv::Mat Twb=(cv::Mat_<float>(4,4)<< cos(theta),-sin(theta),0,x,
                                        sin(theta), cos(theta),0,y,
                                        0,            0,      1, 0,
                                        0,            0,      0, 1);

    return Twb.clone();
}

int Frame::GetBirdMapPointsNum()
{
    int sum = 0;
    for (size_t i = 0; i < mvpMapPointsBird.size(); i++)
    {
        MapPointBird * pMP = mvpMapPointsBird[i];
        if (pMP)
            sum++;        
    }

    return sum;
}

void Frame::CalExTime()
{
    sort(vTimesFront.begin(),vTimesFront.end());
    sort(vTimesBird.begin(),vTimesBird.end());
    double fronttime = 0;
    double birdtime = 0;
    int nImg = vTimesBird.size();
    for(int ni=0; ni<nImg; ni++)
    {
        fronttime+=vTimesFront[ni];
        birdtime+=vTimesBird[ni];
    }
    cout << "-------" << endl << endl;
    cout << "Front median tracking time: " << vTimesFront[nImg/2] << endl;
    cout << "Bird median tracking time: " << vTimesBird[nImg/2] << endl;
    cout << "Front mean tracking time: " << fronttime/nImg << endl;
    cout << "Bird mean tracking time: " << birdtime/nImg << endl;
}

} //namespace ORB_SLAM
