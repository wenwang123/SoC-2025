#include <cmath>
#include <algorithm>
#include <cstdint>
extern "C" {
void poly_add(float* acc,const float* fy,const float* h,int rows,int cols,int stride,int offset,int nt){
#pragma omp parallel for num_threads(nt)
 for(int y=0;y<rows;y++)for(int x=0;x<cols;x++)for(int c=0;c<3;c++){
  float v=fy[y]*h[x*3+c];v+=fy[rows+y]*h[cols*3+x*3+c];v+=fy[rows*2+y]*h[cols*6+x*3+c];acc[(y*stride+offset+x)*3+c]+=v;
 }
}
void rgb_apply(uint8_t* image,const float* poly,const float* coarse,int rows,int cols,int nt){
#pragma omp parallel for num_threads(nt)
 for(int y=0;y<rows;y++)for(int x=0;x<cols*3;x++){
  int i=y*cols*3+x;float f=poly?poly[i]:0.f;if(coarse)f+=coarse[i];float v=float(image[i])/255.f;v+=f;v=std::min(1.f,std::max(0.f,v));v*=255.f;image[i]=uint8_t(std::nearbyintf(v));
 }
}
void pool8(const float* input,float* output,int H,int W,int sy,int sx,int sc,int nt){
 int gh=(H+7)/8,gw=(W+7)/8;
#pragma omp parallel for num_threads(nt)
 for(int gy=0;gy<gh;gy++)for(int gx=0;gx<gw;gx++)for(int c=0;c<3;c++){
  double sum=0.;for(int dy=0;dy<8;dy++){int y=std::min(H-1,gy*8+dy);for(int dx=0;dx<8;dx++){int x=std::min(W-1,gx*8+dx);sum+=input[y*sy+x*sx+c*sc];}}
  output[(gy*gw+gx)*3+c]=float(sum/64.);
 }
}
}
extern "C" void rgb_interp(uint8_t* image,const float* poly,const float* horizontal,const int* y0,const int* y1,const float* ay,int rows,int cols,int nt){
#pragma omp parallel for num_threads(nt)
 for(int y=0;y<rows;y++)for(int x=0;x<cols*3;x++){
  int i=y*cols*3+x;float c=horizontal[y0[y]*cols*3+x]*(1.f-ay[y]);c+=horizontal[y1[y]*cols*3+x]*ay[y];float f=poly?poly[i]:0.f;f+=c;float v=float(image[i])/255.f;v+=f;v=std::min(1.f,std::max(0.f,v));v*=255.f;image[i]=uint8_t(std::nearbyintf(v));
 }
}
