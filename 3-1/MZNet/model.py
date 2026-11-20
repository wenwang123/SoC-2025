"""共享只读权重，用独立Extractor并行计算；融合保持原有顺序。"""
from pathlib import Path
import ctypes,json,os,threading,itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
B=Path(__file__).resolve().parent

class CBranch:
    def __init__(self,prefix,shape=(3,512,512),threads=8,fp16=True):
        self.shape=tuple(shape);self.handle=None;self.closed=False
        meta=Path(prefix).parent/'export.json';info=json.loads(meta.read_text()) if meta.exists() else {};self.required_halo=info.get('required_halo',0);int8=info.get('precision')=='mixed_int8_fp32'
        if int8:fp16=False
        self.lib=ctypes.CDLL(str(B/'native/libesd_ncnn.so'));lib=self.lib
        lib.esd_version.restype=ctypes.c_int;lib.esd_error.restype=ctypes.c_char_p
        lib.esd_create.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_int];lib.esd_create.restype=ctypes.c_void_p
        self.ptr=ctypes.POINTER(ctypes.c_float);lib.esd_forward.argtypes=[ctypes.c_void_p,self.ptr,ctypes.c_int,ctypes.c_int,self.ptr];lib.esd_forward.restype=ctypes.c_int;lib.esd_destroy.argtypes=[ctypes.c_void_p]
        if lib.esd_version()!=20260113:raise RuntimeError('C接口要求NCNN20260113')
        self.handle=lib.esd_create((str(prefix)+'.param').encode(),(str(prefix)+'.bin').encode(),threads,int(fp16),int(int8))
        if not self.handle:raise RuntimeError(lib.esd_error().decode())
    def infer_chw(self,x):
        x=np.ascontiguousarray(x,dtype=np.float32)
        if x.shape!=self.shape:raise ValueError('局部输入必须为RGB512')
        y=np.empty_like(x)
        rc=self.lib.esd_forward(self.handle,x.ctypes.data_as(self.ptr),x.shape[1],x.shape[2],y.ctypes.data_as(self.ptr))
        if rc:raise RuntimeError(self.lib.esd_error().decode())
        if not np.isfinite(y).all():raise RuntimeError('NCNN返回非有限值')
        return y
    def close(self):
        if not self.closed:
            if self.handle:self.lib.esd_destroy(self.handle);self.handle=None
            self.closed=True

class CWorker:
    def __init__(self,parent):
        self.parent=parent;self.shape=parent.shape;self.lib=parent.lib;self.ptr=parent.ptr
        self.lib.esd_worker_create.argtypes=[ctypes.c_void_p];self.lib.esd_worker_create.restype=ctypes.c_void_p
        self.lib.esd_worker_forward.argtypes=[ctypes.c_void_p,self.ptr,ctypes.c_int,ctypes.c_int,self.ptr];self.lib.esd_worker_forward.restype=ctypes.c_int
        self.lib.esd_worker_destroy.argtypes=[ctypes.c_void_p];self.handle=self.lib.esd_worker_create(parent.handle)
        if not self.handle:raise RuntimeError(self.lib.esd_error().decode())
    def infer_chw(self,x):
        x=np.ascontiguousarray(x,dtype=np.float32)
        if x.shape!=self.shape:raise ValueError('局部输入必须为RGB512')
        y=np.empty_like(x);rc=self.lib.esd_worker_forward(self.handle,x.ctypes.data_as(self.ptr),x.shape[1],x.shape[2],y.ctypes.data_as(self.ptr))
        if rc:raise RuntimeError(self.lib.esd_error().decode())
        if not np.isfinite(y).all():raise RuntimeError('NCNN返回非有限值')
        return y
    def close(self):
        if self.handle:self.lib.esd_worker_destroy(self.handle);self.handle=None

class ParallelBranch:
    def __init__(self,prefix,shape=(3,512,512),threads=8,workers=2):
        if threads<workers or threads%workers:raise ValueError('总线程数须能被工作实例数整除')
        self.parent=CBranch(prefix,shape,threads//workers,True);self.required_halo=self.parent.required_halo;self.local=threading.local();self.models=[];self.closed=False;self.workers=workers
        cpus=sorted(os.sched_getaffinity(0))[:threads];counter=itertools.count();barrier=threading.Barrier(workers);per=threads//workers
        def initialize():
            index=next(counter);mask=cpus[index*per:(index+1)*per]
            if len(mask)==per:os.sched_setaffinity(0,set(mask))
            self.local.model=CWorker(self.parent);self.models.append(self.local.model);barrier.wait(timeout=90)
        self.executor=ThreadPoolExecutor(max_workers=workers,initializer=initialize)
        try:
            jobs=[self.executor.submit(lambda:None) for _ in range(workers)]
            for job in jobs:job.result()
        except BaseException:
            self.close();raise
    def _run(self,x):return self.local.model.infer_chw(x)
    def infer_chw(self,x):return self.executor.submit(self._run,x).result()
    def infer_many(self,inputs):
        # 有界预取，避免缓存整幅图的浮点分块；返回顺序保持不变。
        it=iter(inputs);pending=deque()
        def submit():
            try:x=next(it)
            except StopIteration:return False
            pending.append(self.executor.submit(self._run,x));return True
        for _ in range(min(2*self.workers,self.workers+2)):
            if not submit():break
        while pending:
            result=pending.popleft().result();submit();yield result
    def close(self):
        if not self.closed:
            self.executor.shutdown(wait=True,cancel_futures=True)
            for model in self.models:model.close()
            self.parent.close()
            self.closed=True

class Model:
    """给分块融合提供统一接口，CPU线程总数由threads控制。"""
    context='tile'
    preview_calls=0
    def __init__(self,prefix,threads=8,workers=8):
        self.net=ParallelBranch(prefix,threads=threads,workers=workers) if workers>1 else CBranch(prefix,threads=threads)
        self.required_halo=self.net.required_halo;self.native_calls=0
    def prepare_image(self,image):self.native_calls=0
    def infer_chw(self,x):
        y=self.net.infer_chw(x);self.native_calls+=1;return y
    def infer_many(self,patches):
        outputs=self.net.infer_many(patches) if hasattr(self.net,'infer_many') else (self.net.infer_chw(x) for x in patches)
        for y in outputs:self.native_calls+=1;yield y
    def close(self):self.net.close()
