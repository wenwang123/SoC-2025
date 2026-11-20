"""融合计算的本地接口；输入输出约定与Python算法一致。"""
from pathlib import Path
import ctypes,os,numpy as np
lib=ctypes.CDLL(str(Path(__file__).resolve().parent/'native/libesd_ncnn.so'));fp=ctypes.POINTER(ctypes.c_float);bp=ctypes.POINTER(ctypes.c_uint8);nt=min(4,len(os.sched_getaffinity(0)))
lib.poly_add.argtypes=[fp,fp,fp]+[ctypes.c_int]*5;lib.rgb_apply.argtypes=[bp,fp,fp]+[ctypes.c_int]*3;lib.pool8.argtypes=[fp,fp]+[ctypes.c_int]*6
ptr=lambda a:a.ctypes.data_as(fp) if a is not None else None
def poly_add(acc,fy,h,x):lib.poly_add(ptr(acc),ptr(fy),ptr(h),fy.shape[1],h.shape[1],acc.shape[1],x,nt)
def rgb_apply(image,field,coarse=None):lib.rgb_apply(image.ctypes.data_as(bp),ptr(field),ptr(coarse),image.shape[0],image.shape[1],nt)
def pool8(a):
 a=np.asarray(a,dtype=np.float32);h,w=a.shape[:2];out=np.empty(((h+7)//8,(w+7)//8,3),np.float32);lib.pool8(ptr(a),ptr(out),h,w,*[s//4 for s in a.strides],1);return out

lib.rgb_interp.argtypes=[bp,fp,fp,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),fp]+[ctypes.c_int]*3
def rgb_interp(image,field,horizontal,start,height):
    h,w=image.shape[:2];dh=horizontal.shape[0];ys=np.clip((np.arange(start,start+h)+.5)*dh/height-.5,0,dh-1);y0=ys.astype(np.int32);y1=np.minimum(y0+1,dh-1);ay=(ys-y0).astype(np.float32)
    lib.rgb_interp(image.ctypes.data_as(bp),ptr(field),ptr(horizontal),y0.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),y1.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),ptr(ay),h,w,nt)
