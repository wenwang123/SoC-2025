// 共享NCNN只读权重和线程安全内存池，每次推理使用独立Extractor。
#include <net.h>
#include <allocator.h>
#include <mat.h>
#include <c_api.h>
#include <cstring>
#include <string>
#include <memory>
#include <stdexcept>
namespace {
thread_local std::string error;
struct Model {
    ncnn::UnlockedPoolAllocator blobs;
    ncnn::PoolAllocator workspace;
    ncnn::PoolAllocator shared_blobs, shared_workspace;
    ncnn::Net net;
    Model(int threads, bool half, bool int8) {
        blobs.set_size_compare_ratio(0.f);workspace.set_size_compare_ratio(0.f);shared_blobs.set_size_compare_ratio(0.f);shared_workspace.set_size_compare_ratio(0.f);
        net.opt.num_threads=threads;net.opt.openmp_blocktime=0;
        net.opt.lightmode=true;net.opt.use_vulkan_compute=false;
        net.opt.use_fp16_storage=half;net.opt.use_fp16_packed=half;net.opt.use_fp16_arithmetic=half;
        net.opt.use_bf16_storage=false;net.opt.use_bf16_packed=false;
        net.opt.use_int8_inference=int8;net.opt.use_local_pool_allocator=false;
        net.opt.blob_allocator=&blobs;net.opt.workspace_allocator=&workspace;
    }
    void forward(const float* input, int h,int w,float* result) {
        ncnn::Mat data(w,h,3,4u,&blobs);if(data.empty())throw std::runtime_error("input allocation failed");
        const size_t plane=size_t(h)*w;
        for(int c=0;c<3;c++)std::memcpy(data.channel(c),input+c*plane,plane*sizeof(float));
        auto ex=net.create_extractor();
        if(ex.input("in0",data))throw std::runtime_error("input in0 failed");
        ncnn::Mat output;if(ex.extract("out0",output,0))throw std::runtime_error("output out0 failed");
        if(output.w!=w||output.h!=h||output.c!=3||output.elemsize!=4u||output.elempack!=1)throw std::runtime_error("unexpected output format");
        for(int c=0;c<3;c++)std::memcpy(result+c*plane,output.channel(c),plane*sizeof(float));
    }
};
struct Worker {
    Model* model;
    ncnn::UnlockedPoolAllocator blobs;
    ncnn::PoolAllocator workspace;
    Worker(Model* value):model(value){blobs.set_size_compare_ratio(0.f);workspace.set_size_compare_ratio(0.f);}
    void forward(const float* input,int h,int w,float* result){
        ncnn::Mat data(w,h,3,4u,&model->shared_blobs);if(data.empty())throw std::runtime_error("input allocation failed");
        size_t plane=size_t(h)*w;
        for(int c=0;c<3;c++)std::memcpy(data.channel(c),input+c*plane,plane*sizeof(float));
        auto ex=model->net.create_extractor();ex.set_blob_allocator(&model->shared_blobs);ex.set_workspace_allocator(&model->shared_workspace);
        if(ex.input("in0",data))throw std::runtime_error("input in0 failed");
        ncnn::Mat output;if(ex.extract("out0",output,0))throw std::runtime_error("output out0 failed");
        if(output.w!=w||output.h!=h||output.c!=3||output.elemsize!=4u||output.elempack!=1)throw std::runtime_error("unexpected output format");
        for(int c=0;c<3;c++)std::memcpy(result+c*plane,output.channel(c),plane*sizeof(float));
    }
};
}
extern "C" {
int esd_version(){return ncnn_version_number();}
const char* esd_error(){return error.c_str();}
void* esd_create(const char* param,const char* bin,int threads,int half,int int8){
    try{
        error.clear();if(NCNN_VERSION_NUMBER!=20260113||ncnn_version_number()!=20260113)throw std::runtime_error("requires NCNN20260113");
        if(threads<1)throw std::runtime_error("threads must be positive");
        auto m=std::make_unique<Model>(threads,half!=0,int8!=0);
        if(m->net.load_param(param)||m->net.load_model(bin))throw std::runtime_error("model load failed");
        return m.release();
    }catch(const std::exception& e){error=e.what();return nullptr;}
}
int esd_forward(void* model,const float* input,int h,int w,float* result){
    try{error.clear();if(!model||!input||!result||h<32||w<32||h%32||w%32)throw std::runtime_error("invalid input");
        static_cast<Model*>(model)->forward(input,h,w,result);return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
void esd_destroy(void* model){delete static_cast<Model*>(model);}
void esd_set_threads(void* model,int threads){if(model&&threads>0)static_cast<Model*>(model)->net.opt.num_threads=threads;}
void* esd_worker_create(void* model){try{if(!model)throw std::runtime_error("null model");return new Worker(static_cast<Model*>(model));}catch(const std::exception& e){error=e.what();return nullptr;}}
int esd_worker_forward(void* worker,const float* input,int h,int w,float* output){
    try{if(!worker||!input||!output)throw std::runtime_error("null worker or array");static_cast<Worker*>(worker)->forward(input,h,w,output);return 0;}catch(const std::exception& e){error=e.what();return -1;}
}
void esd_worker_destroy(void* worker){delete static_cast<Worker*>(worker);}
}
