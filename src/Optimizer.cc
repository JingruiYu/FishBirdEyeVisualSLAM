﻿/**
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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>
/********************* Modified Here *********************/
#include"OdomG2oTypeSophus.h"
#include "OdomG2oTypeQuat.h"

extern bool bTightCouple;
extern bool bTightCouple2;
extern bool bHaveBird;

namespace ORB_SLAM2
{


void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        unsigned long int id = pMP->mnId+maxKFid+1;

        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if(pKF->mvuRight[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    float infMax = INT_MIN;
    float infMin = INT_MAX;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if (invSigma2 > infMax)
                    infMax = invSigma2;
                if (invSigma2 < infMin)
                    infMin = invSigma2;
                
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    // cout << "inforMax: " << infMax << endl;
    // cout << "inforMin: " << infMin << endl;
    }


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    float frontErrMin = INT_MAX;
    float frontErrMax = INT_MIN;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > frontErrMax)
                frontErrMax = chi2;
            if (chi2 < frontErrMin)
                frontErrMin = chi2;
                
            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    } 
    // cout << "frontErrMax: " << frontErrMax << endl;
    // cout << "frontErrMin: " << frontErrMin << endl;   

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}


int Optimizer::PoseOptimizationWithBird(Frame *pFrame, float wB, float wF)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;
    int nInitialBirdCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    // Set MapPointBird vertices
    const int NB = pFrame->Nbird;

    vector<EdgeSE3ProjectBirdPoint2CamXYZ*> vpEdgesBird;
    vector<size_t> vnIndexEdgeBird;
    vpEdgesBird.reserve(NB);
    vnIndexEdgeBird.reserve(NB);

    const float deltaMono = sqrt(5.991);

    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    float infMax = INT_MIN;
    float infMin = INT_MAX;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2*wF);

                if (invSigma2 > infMax)
                    infMax = invSigma2;
                if (invSigma2 < infMin)
                    infMin = invSigma2;
                
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
                continue;
        }

    }
    // cout << "inforMax: " << infMax << endl;
    // cout << "inforMin: " << infMin << endl;

    for (size_t iB = 0; iB < NB; iB++)
    {
        MapPointBird * pMP = pFrame->mvpMapPointsBird[iB];
        if (pMP)
        {
            Vector3d Xw = Converter::toVector3d(pMP->GetWorldPos());
            Vector3d Xc;
            cv::Point3f p = pFrame->mvKeysBirdCamXYZ[iB];
            Xc << p.x, p.y, p.z;

            EdgeSE3ProjectBirdPoint2CamXYZ *e = new EdgeSE3ProjectBirdPoint2CamXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            const float invSigma2 = pFrame->mvInvLevelSigma2[pFrame->mvKeysBird[iB].octave];
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma2*wB);

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->Xw = Xw;
            e->Xc = Xc;

            optimizer.addEdge(e);
            vpEdgesBird.push_back(e);
            vnIndexEdgeBird.push_back(iB);
            nInitialBirdCorrespondences++;
        }
    }
    }


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={1.5,1.5,1.5,1.5};
    const float chi2Bird[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    int nBadBird=0;
    float frontErrMin = INT_MAX;
    float frontErrMax = INT_MIN;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > frontErrMax)
                frontErrMax = chi2;
            if (chi2 < frontErrMin)
                frontErrMin = chi2;
                
            if(chi2>chi2Mono[it]*(wF+1e-9))
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        nBadBird=0;
        for (size_t iB = 0; iB < vpEdgesBird.size(); iB++)
        {
            EdgeSE3ProjectBirdPoint2CamXYZ * e = vpEdgesBird[iB];

            const size_t idx = vnIndexEdgeBird[iB];

            if (pFrame->mvBirdOutlier[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            float chi2Bad = chi2Bird[it] * (wB+1e-9);
            if (chi2 > chi2Bad)
            {
                pFrame->mvBirdOutlier[idx] = true;
                e->setLevel(1);
                nBadBird++;
            }
            else
            {
                pFrame->mvBirdOutlier[idx] = false;
                e->setLevel(0);
            }
            
            if (it==2)
                e->setRobustKernel(0);
            
        }

        if(optimizer.edges().size()<10)
            break;
    } 


    // cout << "nInitialCorrespondences: " << nInitialCorrespondences << " -nBad: " << nBad << endl;
    // cout << "nInitialBirdCorrespondences: " << nInitialBirdCorrespondences << " -nBadBird: " << nBadBird << endl;  

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}


int Optimizer::BirdOptimization(Frame *pFrame, float wB)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialBirdCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int NB = pFrame->Nbird;

    vector<EdgeSE3ProjectBirdPoint2CamXYZ*> vpEdgesBird;
    vector<size_t> vnIndexEdgeBird;
    vpEdgesBird.reserve(NB);
    vnIndexEdgeBird.reserve(NB);

    const float deltaMono = sqrt(5.991);


    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for (size_t iB = 0; iB < NB; iB++)
    {
        MapPointBird * pMP = pFrame->mvpMapPointsBird[iB];
        if (pMP)
        {
            Vector3d Xw = Converter::toVector3d(pMP->GetWorldPos());
            Vector3d Xc;
            cv::Point3f p = pFrame->mvKeysBirdCamXYZ[iB];
            Xc << p.x, p.y, p.z;

            EdgeSE3ProjectBirdPoint2CamXYZ *e = new EdgeSE3ProjectBirdPoint2CamXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            const float invSigma2 = pFrame->mvInvLevelSigma2[pFrame->mvKeysBird[iB].octave];
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma2*wB);

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->Xw = Xw;
            e->Xc = Xc;

            optimizer.addEdge(e);
            vpEdgesBird.push_back(e);
            vnIndexEdgeBird.push_back(iB);
            nInitialBirdCorrespondences++;
        }
    }
    
    }

    // cout << "\033[35m" << "In optimization, nInitialBirdCorrespondences: " << nInitialBirdCorrespondences << "\033[0m" << endl;

    if(nInitialBirdCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Bird[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};    

    int nBadBird=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBadBird=0;
        
        for (size_t iB = 0; iB < vpEdgesBird.size(); iB++)
        {
            EdgeSE3ProjectBirdPoint2CamXYZ * e = vpEdgesBird[iB];

            const size_t idx = vnIndexEdgeBird[iB];

            if (pFrame->mvBirdOutlier[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            float chi2Bad = chi2Bird[it] * (wB+1e-9);
            if (chi2 > chi2Bad)
            {
                pFrame->mvBirdOutlier[idx] = true;
                e->setLevel(1);
                nBadBird++;
            }
            else
            {
                pFrame->mvBirdOutlier[idx] = false;
                e->setLevel(0);
            }
            
            if (it==2)
                e->setRobustKernel(0);
            
        }
        
        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    // cout << "In optimization, nBadBird: " << nBadBird << endl;

    return nInitialBirdCorrespondences-nBadBird;
}


void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        // vSE3->setFixed(pKFi->mnId==0);
        vSE3->setFixed(pKFi->isInit);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        unsigned long int id = pMP->mnId+maxKFid+1;

        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {

    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    // Optimize again without the outliers

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    cout << "OptimizeEssentialGraph" << endl;
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    const vector<MapPointBird*> vpMPBs = pMap->GetAllMapPointsBird();

    int lessObsPB = 0;
    for (size_t i = 0; i < vpMPBs.size(); i++)
    {
        MapPointBird* pMPB = vpMPBs[i];
        if(pMPB->Observations()<2)
            lessObsPB++;
    }

    cout << "\033[1m\033[33m" << "all MPB size is " << vpMPBs.size() << " , the number of One obser is : " << lessObsPB << "\033[0m" << endl;
    


    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }
        else
        {
            cout << "\033[1m\033[33m" << "there are KF without parent, pKF->mnId: " << pKF->mnId << ",pKF->mnFrameId: " << pKF->mnFrameId << "\033[0m" << endl;
        }
        

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }

        if (vpConnectedKFs.empty())
        {
            cout << "vpConnectedKFs.empty() exist." << endl;
            const vector<KeyFrame*> vpConnectedBirdKFs = pKF->GetBirdConnectedBirdKeyFrames();
            cout << "vpConnectedBirdKFs.empty() size is: " << vpConnectedBirdKFs.size() << endl;
            for (vector<KeyFrame*>::const_iterator vit=vpConnectedBirdKFs.begin(); vit!=vpConnectedBirdKFs.end(); vit++)
            {
                KeyFrame* pKFn = *vit;
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    if (optimizer.edges().size() < 1)
    {
        cout << "\033[1m\033[33m" << "before OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.edges().size() << endl;
        cout << "LoopConnections.size(): " << LoopConnections.size() << endl;
        cout << "vpKFs.size(): " << vpKFs.size() << endl; 
        
    }
    
    if (optimizer.vertices().size() < 1)
    {
        cout << "\033[1m\033[33m" << "before OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.vertices().size() << endl;
    }

    // Optimize!
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    if (optimizer.edges().size() < 1)
    {
        cout << "\033[1m\033[33m" << "after OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.edges().size() << endl;
    }
    
    if (optimizer.vertices().size() < 1)
    {
        cout << "\033[1m\033[33m" << "after OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.vertices().size() << endl;
    }
    

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    int mpRfKF = 0;
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            if (!pRefKF)
                mpRfKF++;
            
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
    cout << "mpRfKF empty number: " << mpRfKF << endl;

    int mpRfBKF=0;
    int RfBKFNe=0;
    for (size_t i = 0; i < vpMPBs.size(); i++)
    {
        MapPointBird* pMPB = vpMPBs[i];

        if (pMPB->isBad())
            continue;

        int nIDr;
        if(pMPB->mnCorrectedByKF==pCurKF->mnId) 
        {
            nIDr = pMPB->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMPB->GetReferenceKeyFrame();
            if (!pRefKF)
            {
                mpRfBKF++;
                continue;
            }
            else
            {
                RfBKFNe++;
            }
            
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMPB->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMPB->SetWorldPos(cvCorrectedP3Dw);       
    }
    cout << "mpRfBKF empty number: " << mpRfBKF << ", RfBKFNe not empty number: " << RfBKFNe << endl;
    
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    if (optimizer.edges().size() < 1)
    {
        cout << "\033[1m\033[33m" << "before OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.edges().size() << endl;
    }
    
    if (optimizer.vertices().size() < 1)
    {
        cout << "\033[1m\033[33m" << "before OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.vertices().size() << endl;
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    if (optimizer.edges().size() < 1)
    {
        cout << "\033[1m\033[33m" << "after OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.edges().size() << endl;
    }
    
    if (optimizer.vertices().size() < 1)
    {
        cout << "\033[1m\033[33m" << "after OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.vertices().size() << endl;
    }

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}

/********************* Modified Here *********************/
void Optimizer::GlobalBundleAdjustemntWithOdom(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    vector<MapPointBird*> vpMPB = pMap->GetAllMapPointsBird();
    // cout << "\033[31m" << "in GlobalBundleAdjustemntWithOdom, vpMPB.size(): " << vpMPB.size() << "\033[0m" << endl;
    BundleAdjustmentWithOdom(vpKFs,vpMP,vpMPB,nIterations,pbStopFlag, nLoopKF, bRobust);
}

void Optimizer::BundleAdjustmentWithOdom(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, const vector<MapPointBird *> &vpMPB,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust, const float wF, const float wB, const float wP)
{    
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    vector<bool> vbNotIncludedMPBird;
    vbNotIncludedMPBird.resize(vpMPB.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    vector<pair<int,KeyFrame*>> vKFvertices;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
#ifdef USE_SOPHUS
        VertexSE3LieAlgebra * vSE3 = new VertexSE3LieAlgebra();
        vSE3->setEstimate(SE3(
            Converter::toMatrix3d(pKF->GetRotation()),
            Converter::toVector3d(pKF->GetTranslation())));
#else
        VertexSE3Quat * vSE3 = new VertexSE3Quat();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
#endif
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        // vKFvertices.push_back(pKF);
        vKFvertices.push_back(make_pair(pKF->mnId,pKF));
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
        // std::cout<<"KF: "<<pKF->mnId<<" added."<<endl;
    }

    const float thHuber2D = sqrt(5.99);
    long unsigned int maxMPid = 0;
    float frontErrMin = INT_MAX;
    float frontErrMax = INT_MIN;
    float inforMin = INT_MAX;
    float inforMax = INT_MIN;
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        unsigned long int id = pMP->mnId+maxKFid+1;
        if(id>maxMPid)
            maxMPid = id;

        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if(pKF->mvuRight[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;
#ifdef USE_SOPHUS
                EdgeProjectXYZ2UVLieAlgebra* e = new EdgeProjectXYZ2UVLieAlgebra();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
#else
                EdgeSE3ProjectXYZ2UVQuat * e = new EdgeSE3ProjectXYZ2UVQuat();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
#endif
                
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2*wF);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
                // vpEdgesBA.push_back(e);

                {
                    e->computeError();
                    float chi2 = e->chi2();
                    if (chi2 > frontErrMax)
                        frontErrMax = chi2;
                    if (chi2 < frontErrMin)
                        frontErrMin = chi2;

                    if (invSigma2 > inforMax)
                        inforMax = invSigma2;
                    if (invSigma2 < inforMin)
                        inforMin = invSigma2;
                    
                }
            }
            else
            {
                continue;
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }
    cout << "frontErrMax: " << frontErrMax << endl;
    cout << "frontErrMin: " << frontErrMin << endl;
    cout << "inforMax: " << inforMax << endl;
    cout << "inforMin: " << inforMin << endl;
    
    // Set MapPointBird vertices
    for (size_t i = 0; i < vpMPB.size(); i++)
    {
        MapPointBird* pMPB = vpMPB[i];
        if (pMPB->isBad() || !bHaveBird)
            continue;
        
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMPB->GetWorldPos()));
        unsigned long int id = pMPB->mnId+maxMPid+1;

        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMPB->GetObservations(); //

        int nEdgesBird = 0;
        for (map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;   

            nEdgesBird++;

            const cv::Point3f pt = pKF->mvKeysBirdCamXYZ[mit->second];         

            Eigen::Matrix<double,3,1> obs;
            obs << pt.x, pt.y, pt.z;

            EdgeSE3ProjectXYZ2XYZQuat *e = new EdgeSE3ProjectXYZ2XYZQuat();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(obs);
            const float &invSigma2 = pKF->mvInvLevelSigma2[pKF->mvKeysBird[mit->second].octave];
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma2*wB);

            if (bRobust)
            {
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);
            }
            
            optimizer.addEdge(e);
        }

        if (nEdgesBird == 0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMPBird[i] = true;
        }
        else
        {
            vbNotIncludedMPBird[i] = false;
        } 
    }
    

//     //PoseGraph Constraint
//     sort(vKFvertices.begin(),vKFvertices.end());
//     // for(vector<KeyFrame*>::iterator KFit=vKFvertices.begin(),KFnext=KFit+1;KFnext!=vKFvertices.end();KFit++,KFnext++)
//     for(vector<pair<int,KeyFrame*>>::iterator KFit=vKFvertices.begin(),KFnext=KFit+1;KFnext!=vKFvertices.end();KFit++,KFnext++)
//     {
//         // KeyFrame *KF1=*KFit;
//         // KeyFrame *KF2=*KFnext;
//         KeyFrame *KF1=KFit->second;
//         KeyFrame *KF2=KFnext->second;
//         Eigen::Matrix4d T12=Converter::toMatrix4d(Frame::GetTransformFromOdometer(KF1->mGtPose,KF2->mGtPose));
// #ifdef USE_SOPHUS
//         EdgeSE3LieAlgebra *e=new EdgeSE3LieAlgebra();
//         e->setMeasurement(SE3(T12.block(0,0,3,3),T12.block(0,3,3,1)));
// #else
//         EdgeSE3Quat *e = new EdgeSE3Quat();
//         e->setMeasurement(g2o::SE3Quat(T12.block(0,0,3,3),T12.block(0,3,3,1)));
// #endif
//         e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(KF1->mnId)));
//         e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(KF2->mnId)));
        
//         Eigen::Matrix3d Info1=Eigen::Matrix3d::Identity()*1e3*wP;
//         Eigen::Matrix3d Info2=Eigen::Matrix3d::Identity()*1e3*wP;
//         g2o::Matrix6d Info; 
//         Info<<Info1,Eigen::Matrix3d::Zero(),Eigen::Matrix3d::Zero(),Info2;
//         e->setInformation(Info);
//         //e->setInformation(g2o::Matrix6d::Identity()*10000);

//         // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//         // e->setRobustKernel(rk);
//         // rk->setDelta(thHuber6D);

//         optimizer.addEdge(e);
//         cout<<"add pose constraint between KF ("<<KF1->mnId<<" , "<<KF2->mnId<<")"<<endl;
//     }


    if (optimizer.edges().size() < 1)
    {
        cout << "\033[1m\033[33m" << "before OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.edges().size() << endl;
    }
    
    if (optimizer.vertices().size() < 1)
    {
        cout << "\033[1m\033[33m" << "before OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.vertices().size() << endl;
    }

    // Optimize!
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    
    if (optimizer.edges().size() < 1)
    {
        cout << "\033[1m\033[33m" << "after OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.edges().size() << endl;
    }
    
    if (optimizer.vertices().size() < 1)
    {
        cout << "\033[1m\033[33m" << "after OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.vertices().size() << endl;
    }

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
#ifdef USE_SOPHUS
        VertexSE3LieAlgebra* vSE3 = static_cast<VertexSE3LieAlgebra*>(optimizer.vertex(pKF->mnId));
        SE3 Tcwse3=vSE3->estimate();
        Eigen::Matrix4d Tcw;
        Tcw<<Tcwse3.rotation_matrix(),Tcwse3.translation(),Eigen::Vector3d::Zero().transpose(),1;
#else
        VertexSE3Quat *vSE3 = static_cast<VertexSE3Quat*>(optimizer.vertex(pKF->mnId));
        Eigen::Matrix4d Tcw = vSE3->estimate().to_homogeneous_matrix();
#endif
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(Tcw));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(Tcw).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

    // Birdview Points
    for (size_t i = 0; i < vpMPB.size(); i++)
    {
        if (vbNotIncludedMPBird[i] || !bHaveBird)
            continue;

        MapPointBird* pMPB = vpMPB[i];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMPB->mnId+maxMPid+1));
        if (nLoopKF==0)
        {
            pMPB->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        }
        else
        {
            pMPB->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMPB->mPosGBA);
            pMPB->mnBAGlobalForKF = nLoopKF;
        }
    }    
}

void Optimizer::LocalBundleAdjustmentWithOdom(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, const float wF, const float wB, const float wP)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    list<MapPointBird*> lLocalMapPointsBirds;
    if (bHaveBird)
    {
        for (list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            vector<MapPointBird*> vpMPBs = (*lit)->GetMapPointBirdMatches();
            for (vector<MapPointBird*>::iterator vit=vpMPBs.begin(), vend=vpMPBs.end(); vit!=vend; vit++)
            {
                MapPointBird* pMPB = *vit;
                if (pMPB)
                {
                    if (!pMPB->isBad())
                    {
                        if (pMPB->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapPointsBirds.push_back(pMPB);
                            pMPB->mnBALocalForKF=pKF->mnId;
                        }
                    }
                }
            }
        }

        for(list<MapPointBird*>::iterator lit=lLocalMapPointsBirds.begin(), lend=lLocalMapPointsBirds.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
            for (map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;
                if (pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
#ifdef USE_SOPHUS
        VertexSE3LieAlgebra * vSE3 = new VertexSE3LieAlgebra();
        vSE3->setEstimate(SE3(
            Converter::toMatrix3d(pKFi->GetRotation()),
            Converter::toVector3d(pKFi->GetTranslation())));
#else
        VertexSE3Quat *vSE3 = new VertexSE3Quat();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
#endif
        vSE3->setId(pKFi->mnId);
        // vSE3->setFixed(pKFi->mnId==0);
        vSE3->setFixed(pKFi->isInit);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
#ifdef USE_SOPHUS
        VertexSE3LieAlgebra * vSE3 = new VertexSE3LieAlgebra();
        vSE3->setEstimate(SE3(
            Converter::toMatrix3d(pKFi->GetRotation()),
            Converter::toVector3d(pKFi->GetTranslation())));
#else
        VertexSE3Quat *vSE3 = new VertexSE3Quat();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
#endif
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();
    const int nExpectedBirdSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPointsBirds.size();

#ifdef USE_SOPHUS
    vector<EdgeProjectXYZ2UVLieAlgebra*> vpEdgesMono;
#else
    vector<EdgeSE3ProjectXYZ2UVQuat*> vpEdgesMono;
#endif
    vpEdgesMono.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<EdgeSE3ProjectXYZ2XYZQuat*> vpEdgesBird;
    vpEdgesBird.reserve(nExpectedBirdSize);
    vector<KeyFrame*> vpEdgeKFBird;
    vpEdgesBird.reserve(nExpectedBirdSize);
    vector<MapPointBird*> vpMapPointEdgeBird;
    vpMapPointEdgeBird.reserve(nExpectedBirdSize);    

    const float thHuberMono = sqrt(5.991);
    // const float thHuberStereo = sqrt(7.815);
    // const float thHuber6D = sqrt(12.59);
    long unsigned int maxMPid = 0;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        unsigned long int id = pMP->mnId+maxKFid+1;
        if (id > maxMPid)
            maxMPid = id;
        
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;
#ifdef USE_SOPHUS
                    EdgeProjectXYZ2UVLieAlgebra* e = new EdgeProjectXYZ2UVLieAlgebra();
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
#else
                    EdgeSE3ProjectXYZ2UVQuat *e = new EdgeSE3ProjectXYZ2UVQuat();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
#endif
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2*wF);
                    // std::cout<<"invSigma2 = "<<invSigma2<<endl;

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    continue;
                }
            }
        }
    }

    // Birdview Points and Edges
    for (list<MapPointBird*>::iterator lit=lLocalMapPointsBirds.begin(), lend=lLocalMapPointsBirds.end(); lit!=lend; lit++)
    {
        MapPointBird* pMPBird = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMPBird->GetWorldPos()));
        int id = pMPBird->mnId+maxMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMPBird->GetObservations();
        for (map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            if (!pKFi->isBad())
            {
                const cv::Point3f &pt = pKFi->mvKeysBirdCamXYZ[mit->second];

                Eigen::Matrix<double,3,1> obs;
                obs << pt.x, pt.y, pt.z;

                EdgeSE3ProjectXYZ2XYZQuat *e = new EdgeSE3ProjectXYZ2XYZQuat();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKFi->mvInvLevelSigma2[pKFi->mvKeysBird[mit->second].octave];
                e->setInformation(Eigen::Matrix3d::Identity()*invSigma2*wB);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                optimizer.addEdge(e);
                vpEdgesBird.push_back(e);
                vpEdgeKFBird.push_back(pKFi);
                vpMapPointEdgeBird.push_back(pMPBird);
            }
        }
    }
    
    if (bTightCouple)
    {    
        //PoseGraph Constraints
        vector<KeyFrame*> vLocalKeyFrames(lLocalKeyFrames.begin(),lLocalKeyFrames.end());
        sort(vLocalKeyFrames.begin(),vLocalKeyFrames.end(),[](KeyFrame* a,KeyFrame* b){return (a->mnId<b->mnId);});
        for(vector<KeyFrame*>::iterator KFit=vLocalKeyFrames.begin(),KFnext=KFit+1;KFnext!=vLocalKeyFrames.end();KFit++,KFnext++)
        {
            KeyFrame *KF1=*KFit;
            KeyFrame *KF2=*KFnext;
            Eigen::Matrix4d T12=Converter::toMatrix4d(Frame::GetTransformFromOdometer(KF1->mGtPose,KF2->mGtPose));
    #ifdef USE_SOPHUS
            EdgeSE3LieAlgebra *e=new EdgeSE3LieAlgebra();
            e->setMeasurement(SE3(T12.block(0,0,3,3),T12.block(0,3,3,1)));
    #else
            EdgeSE3Quat *e = new EdgeSE3Quat();
            e->setMeasurement(g2o::SE3Quat(T12.block(0,0,3,3),T12.block(0,3,3,1)));
    #endif
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(KF1->mnId)));
            e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(KF2->mnId)));
            
            // Eigen::Matrix3d Info1=Eigen::Matrix3d::Identity()*5e2;
            // Eigen::Matrix3d Info2=Eigen::Matrix3d::Identity()*5e1;
            // g2o::Matrix6d Info;
            // Info<<Info1,Eigen::Matrix3d::Zero(),Eigen::Matrix3d::Zero(),Info2;
            g2o::Matrix6d Info=g2o::Matrix6d::Identity()*1e4;
            e->setInformation(Info*wP);
            
            // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            // e->setRobustKernel(rk);
            // rk->setDelta(thHuber6D);

            optimizer.addEdge(e);
            // cout<<"add pose constraint between KF ("<<KF1->mnId<<" , "<<KF2->mnId<<")"<<endl;

            if (bTightCouple2)
            {
                vector<KeyFrame*>::iterator KFextra=KFnext+1;
                if(KFextra!=vLocalKeyFrames.end())
                {
                    KeyFrame *KF3=*KFextra;
                    Eigen::Matrix4d T13=Converter::toMatrix4d(Frame::GetTransformFromOdometer(KF1->mGtPose,KF3->mGtPose));
#ifdef USE_SOPHUS
                    EdgeSE3LieAlgebra *e=new EdgeSE3LieAlgebra();
                    e->setMeasurement(SE3(T13.block(0,0,3,3),T13.block(0,3,3,1)));
#else
                    EdgeSE3Quat *e = new EdgeSE3Quat();
                    e->setMeasurement(g2o::SE3Quat(T13.block(0,0,3,3),T13.block(0,3,3,1)));
#endif
                    e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(KF1->mnId)));
                    e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(KF3->mnId)));
                    
                    e->setInformation(g2o::Matrix6d::Identity()*2e3);
                    optimizer.addEdge(e);
                    // cout<<"add extra pose constraint between KF ("<<KF1->mnId<<" , "<<KF3->mnId<<")"<<endl;

                    vector<KeyFrame*>::iterator KFextra2=KFnext+2;
                    if(KFextra2!=vLocalKeyFrames.end())
                    {
                        KeyFrame *KF4=*KFextra2;
                        Eigen::Matrix4d T14=Converter::toMatrix4d(Frame::GetTransformFromOdometer(KF1->mGtPose,KF4->mGtPose));
#ifdef USE_SOPHUS
                        EdgeSE3LieAlgebra *e=new EdgeSE3LieAlgebra();
                        e->setMeasurement(SE3(T14.block(0,0,3,3),T14.block(0,3,3,1)));
#else
                        EdgeSE3Quat *e = new EdgeSE3Quat();
                        e->setMeasurement(g2o::SE3Quat(T14.block(0,0,3,3),T14.block(0,3,3,1)));
#endif
                        e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(KF1->mnId)));
                        e->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(KF4->mnId)));
                        e->setInformation(g2o::Matrix6d::Identity()*1e3*wP);
                        optimizer.addEdge(e);
                        // cout<<"add 2nd extra pose constraint between KF ("<<KF1->mnId<<" , "<<KF4->mnId<<")"<<endl;
                    }
                }
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    if (optimizer.edges().size() < 1)
    {
        cout << "\033[1m\033[33m" << "before OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.edges().size() << endl;
    }
    
    if (optimizer.vertices().size() < 1)
    {
        cout << "\033[1m\033[33m" << "before OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.vertices().size() << endl;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    if (optimizer.edges().size() < 1)
    {
        cout << "\033[1m\033[33m" << "after OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.edges().size() << endl;
    }
    
    if (optimizer.vertices().size() < 1)
    {
        cout << "\033[1m\033[33m" << "after OptimizeEssentialGraph : optimizer.edges().size() < 2: " << optimizer.vertices().size() << endl;
    }

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {

    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
#ifdef USE_SOPHUS
        EdgeProjectXYZ2UVLieAlgebra* e = vpEdgesMono[i];
#else
        EdgeSE3ProjectXYZ2UVQuat *e = vpEdgesMono[i];
#endif
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    for (size_t i = 0, iend=vpEdgesBird.size(); i < iend; i++)
    {
        EdgeSE3ProjectXYZ2XYZQuat *e = vpEdgesBird[i];
        
        e->computeError();
        if(e->chi2()>5.991) 
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }
    

    // Optimize again without the outliers
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    // vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    vToErase.reserve(vpEdgesMono.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
#ifdef USE_SOPHUS
        EdgeProjectXYZ2UVLieAlgebra* e = vpEdgesMono[i];
#else
        EdgeSE3ProjectXYZ2UVQuat *e = vpEdgesMono[i];
#endif
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    vector< pair<KeyFrame*,MapPointBird*> > vToEraseBird;
    vToEraseBird.reserve(vpEdgesBird.size());
    for (size_t i = 0; i < vpEdgesBird.size(); i++)
    {
        EdgeSE3ProjectXYZ2XYZQuat *e = vpEdgesBird[i];
        
        if(e->chi2()>5.991)
        {
            MapPointBird* pMPBird = vpMapPointEdgeBird[i];
            KeyFrame* pKFi = vpEdgeKFBird[i];
            vToEraseBird.push_back(make_pair(pKFi,pMPBird));
        }
    }
    
    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    if (!vToEraseBird.empty())
    {
        for (size_t i = 0; i < vToEraseBird.size(); i++)
        {
            KeyFrame* pKFi = vToEraseBird[i].first;
            MapPointBird* pMPi = vToEraseBird[i].second;
            pKFi->EraseMapPointBirdMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    
    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
#ifdef USE_SOPHUS
        VertexSE3LieAlgebra* vSE3 = static_cast<VertexSE3LieAlgebra*>(optimizer.vertex(pKF->mnId));
        SE3 Tcwse3=vSE3->estimate();
        Eigen::Matrix4d Tcw;
        Tcw<<Tcwse3.rotation_matrix(),Tcwse3.translation(),Eigen::Vector3d::Zero().transpose(),1;
#else
        VertexSE3Quat *vSE3 = static_cast<VertexSE3Quat*>(optimizer.vertex(pKF->mnId));
        Eigen::Matrix4d Tcw = vSE3->estimate().to_homogeneous_matrix();
#endif
        pKF->SetPose(Converter::toCvMat(Tcw));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    for(list<MapPointBird*>::iterator lit=lLocalMapPointsBirds.begin(), lend=lLocalMapPointsBirds.end(); lit!=lend; lit++)
    {
        MapPointBird* pMPB = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMPB->mnId+maxMPid+1));
        pMPB->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    }
}


} //namespace ORB_SLAM
