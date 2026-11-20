"""推理入口：原生分辨率、512分块，输出无损PNG。"""
import os,sys,time
sys.dont_write_bytecode=True
if __name__=='__main__' and 'MALLOC_ARENA_MAX' not in os.environ:
    os.execve(sys.executable,[sys.executable,*sys.argv],{**os.environ,'MALLOC_ARENA_MAX':'2','MALLOC_MMAP_THRESHOLD_':'131072','MALLOC_TRIM_THRESHOLD_':'262144','OPENBLAS_NUM_THREADS':'1'})
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,struct,zlib,binascii
import numpy as np
from PIL import Image
from model import Model
from tiling import TileConfig
from fusion import restore_consistent

def write_stored_png(path,image):
    """按128行写入，控制额外内存；文件约为原始RGB数据大小。"""
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('只接受H×W×3的uint8 RGB图像')
    h,w=image.shape[:2];rows=np.empty((128,w*3+1),dtype=np.uint8);rows[:,0]=0
    with Path(path).open('wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        def chunk(kind,data):
            f.write(struct.pack('>I',len(data)));f.write(kind);f.write(data)
            f.write(struct.pack('>I',binascii.crc32(data,binascii.crc32(kind))&0xffffffff))
        chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0));compressor=zlib.compressobj(level=0)
        for start in range(0,h,128):
            n=min(128,h-start);rows[:n,1:]=image[start:start+n].reshape(n,w*3)
            data=compressor.compress(rows[:n].tobytes())
            if data:chunk(b'IDAT',data)
        data=compressor.flush()
        if data:chunk(b'IDAT',data)
        chunk(b'IEND',b'')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path,help='输入照片')
    parser.add_argument('--output',required=True,type=Path,help='结果PNG路径')
    parser.add_argument('--threads',type=int,default=8,help='CPU线程总数')
    parser.add_argument('--workers',type=int,help='并行分块数，默认按尺寸选择')
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve() or args.output.suffix.lower()!='.png':parser.error('输出需要是另一个PNG文件')
    if args.threads<1:parser.error('线程数必须大于0')
    allowed=sorted(os.sched_getaffinity(0));args.threads=min(args.threads,len(allowed))
    os.sched_setaffinity(0,set(allowed[:args.threads]))
    t=time.perf_counter();image=np.asarray(Image.open(args.input).convert('RGB'))
    # 大于4K的照片减少并发，给整幅RGB结果和融合缓冲留出内存。
    cap=8 if image.shape[0]*image.shape[1]<=3840*2160 else 4
    workers=args.workers or max(n for n in range(1,min(cap,args.threads)+1) if args.threads%n==0)
    if workers<1 or args.threads%workers:parser.error('线程总数需要能被并行分块数整除')
    model=Model(Path(__file__).resolve().parent/'models/model',args.threads,workers)
    try:
        config=TileConfig(480,32,16,8)
        if config.halo<model.required_halo:raise ValueError('模型要求更大的上下文边缘')
        result,report=restore_consistent(model,image,assembly='stable',config=config)
        args.output.parent.mkdir(parents=True,exist_ok=True);write_stored_png(args.output,result)
        print(json.dumps({'output':str(args.output),'run_seconds':time.perf_counter()-t,'cpu_threads':args.threads,'workers':workers,'native_calls':model.native_calls,'ncnn':'20260113'},ensure_ascii=False))
    finally:model.close()

if __name__=='__main__':main()
