"""Check whether a court fits a plausible camera and project its net."""


import cv2
import numpy as np

from .geometry import CORNER_COURT_M


def camera(corners, size):
    width,height=size
    transform=cv2.getPerspectiveTransform(CORNER_COURT_M,np.asarray(corners,np.float32))
    focal=np.geomspace(.4*width,4*width,200)
    axes=np.broadcast_to(transform[:,:2],(len(focal),3,2)).copy()
    axes[:,:2]-=np.array([width/2,height/2])[None,:,None]*axes[:,2:3]
    axes[:,:2]/=focal[:,None,None]
    norms=np.linalg.norm(axes,axis=1)
    cosine=(axes[:,:,0]*axes[:,:,1]).sum(axis=1)/np.prod(norms,axis=1)
    ratio=np.log(norms[:,0]/norms[:,1])
    error=np.hypot(cosine,ratio)
    best=error.argmin()
    intrinsic=np.array([[focal[best],0,width/2],[0,focal[best],height/2],[0,0,1]])
    pose=np.linalg.inv(intrinsic)@transform
    pose/=np.linalg.norm(pose[:,:2],axis=0).mean()
    vertical=np.cross(pose[:,0],pose[:,1])
    vertical/=np.linalg.norm(vertical)
    return float(error[best]),float(focal[best]/width),pose,intrinsic,vertical


def net_segments(corners, size):
    error,focal,pose,intrinsic,vertical=camera(corners,size)
    # The court y axis points towards the camera; the cross-product normal points down.
    ground=np.array([[0,6.7,1],[3.05,6.7,1],[6.1,6.7,1]])@pose.T
    top=ground-np.array([1.55,1.524,1.55])[:,None]*vertical
    ground_px=ground@intrinsic.T
    top_px=top@intrinsic.T
    ground_px=ground_px[:,:2]/ground_px[:,2:]
    top_px=top_px[:,:2]/top_px[:,2:]
    segments=np.array([[top_px[0],top_px[1]],[top_px[1],top_px[2]],
                       [ground_px[0],top_px[0]],[ground_px[2],top_px[2]]])
    return segments,error,focal
