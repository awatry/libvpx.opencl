/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef DYNAMIC_CL_H
#define	DYNAMIC_CL_H

#ifdef	__cplusplus
extern "C" {
#endif

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
    
#include <dlfcn.h>

int load_cl(char *lib_name);
int close_cl();

typedef cl_int(*fn_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int(*fn_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
typedef cl_int(*fn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int(*fn_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);
typedef cl_context(*fn_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *, void (*pfn_notify)(const char *, const void *, size_t, void *), void *, cl_int *);
typedef cl_context(*fn_clCreateContextFromType)(const cl_context_properties *, cl_device_type, void (*pfn_notify)(const char *, const void *, size_t, void *), void *, cl_int *);
typedef cl_int(*fn_clRetainContext)(cl_context);
typedef cl_int(*fn_clReleaseContext)(cl_context);
typedef cl_int(*fn_clGetContextInfo)(cl_context, cl_context_info, size_t, void *, size_t *);
typedef cl_command_queue(*fn_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
typedef cl_int(*fn_clRetainCommandQueue)(cl_command_queue);
typedef cl_int(*fn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int(*fn_clGetCommandQueueInfo)(cl_command_queue, cl_command_queue_info, size_t, void *, size_t *);
typedef cl_mem(*fn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_mem(*fn_clCreateImage2D)(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, void *, cl_int *);
typedef cl_mem(*fn_clCreateImage3D)(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, size_t, size_t, void *, cl_int *);
typedef cl_int(*fn_clRetainMemObject)(cl_mem);
typedef cl_int(*fn_clReleaseMemObject)(cl_mem);
typedef cl_int(*fn_clGetSupportedImageFormats)(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format *, cl_uint *);
typedef cl_int(*fn_clGetMemObjectInfo)(cl_mem, cl_mem_info, size_t, void *, size_t *);
typedef cl_int(*fn_clGetImageInfo)(cl_mem, cl_image_info, size_t, void *, size_t *);
typedef cl_sampler(*fn_clCreateSampler)(cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, cl_int *);
typedef cl_int(*fn_clRetainSampler)(cl_sampler);
typedef cl_int(*fn_clReleaseSampler)(cl_sampler);
typedef cl_int(*fn_clGetSamplerInfo)(cl_sampler, cl_sampler_info, size_t, void *, size_t *);
typedef cl_program(*fn_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_program(*fn_clCreateProgramWithBinary)(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *);
typedef cl_int(*fn_clRetainProgram)(cl_program);
typedef cl_int(*fn_clReleaseProgram)(cl_program);
typedef cl_int(*fn_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,  void (*pfn_notify)(cl_program,void*), void *);
typedef cl_int(*fn_clUnloadCompiler)(void);
typedef cl_int(*fn_clGetProgramInfo)(cl_program, cl_program_info, size_t, void *, size_t *);
typedef cl_int(*fn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
typedef cl_kernel(*fn_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int(*fn_clCreateKernelsInProgram)(cl_program, cl_uint, cl_kernel *, cl_uint *);
typedef cl_int(*fn_clRetainKernel)(cl_kernel);
typedef cl_int(*fn_clReleaseKernel)(cl_kernel);
typedef cl_int(*fn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int(*fn_clGetKernelInfo)(cl_kernel, cl_kernel_info, size_t, void *, size_t *);
typedef cl_int(*fn_clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);
typedef cl_int(*fn_clWaitForEvents)(cl_uint, const cl_event *);
typedef cl_int(*fn_clGetEventInfo)(cl_event, cl_event_info, size_t, void *, size_t *);
typedef cl_int(*fn_clRetainEvent)(cl_event);
typedef cl_int(*fn_clReleaseEvent)(cl_event);
typedef cl_int(*fn_clGetEventProfilingInfo)(cl_event, cl_profiling_info, size_t, void *, size_t *);
typedef cl_int(*fn_clFlush)(cl_command_queue);
typedef cl_int(*fn_clFinish)(cl_command_queue);
typedef cl_int(*fn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueWriteBuffer)(cl_command_queue,  cl_mem,  cl_bool,  size_t,  size_t,  const void *,  cl_uint,  const cl_event *,  cl_event *);
typedef cl_int(*fn_clEnqueueCopyBuffer)(cl_command_queue,  cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueReadImage)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueWriteImage)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueCopyImage)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueCopyImageToBuffer)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, size_t, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueCopyBufferToImage)(cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef void*(*fn_clEnqueueMapBuffer)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event *, cl_event *, cl_int *);
typedef void*(*fn_clEnqueueMapImage)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t *, const size_t *, size_t *, size_t *, cl_uint, const cl_event *, cl_event *, cl_int *);
typedef cl_int(*fn_clEnqueueUnmapMemObject)(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueTask)(cl_command_queue, cl_kernel, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueNativeKernel)(cl_command_queue,					 void (*user_func)(void *), void *, size_t, cl_uint, const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *);
typedef cl_int(*fn_clEnqueueMarker)(cl_command_queue, cl_event *);
typedef cl_int(*fn_clEnqueueWaitForEvents)(cl_command_queue, cl_uint, const cl_event *);
typedef cl_int(*fn_clEnqueueBarrier)(cl_command_queue);
typedef void*(*fn_clGetExtensionFunctionAddress)(const char *);

typedef struct CL_FUNCTIONS {
    fn_clGetPlatformIDs getPlatformIDs;
    fn_clGetPlatformInfo getPlatformInfo;
    fn_clGetDeviceIDs getDeviceIDs;
    fn_clGetDeviceInfo getDeviceInfo;
    fn_clCreateContext createContext;
    fn_clCreateContextFromType createContextFromType;
    fn_clRetainContext retainContext;
    fn_clReleaseContext releaseContext;
    fn_clGetContextInfo getContextInfo;
    fn_clCreateCommandQueue createCommandQueue;
    fn_clRetainCommandQueue retainCommandQueue;
    fn_clReleaseCommandQueue releaseCommandQueue;
    fn_clGetCommandQueueInfo getCommandQueue;
    fn_clCreateBuffer createBuffer;
    fn_clCreateImage2D createImage2D;
    fn_clCreateImage3D createImage3D;
    fn_clRetainMemObject retainMemObject;
    fn_clReleaseMemObject releaseMemObject;
    fn_clGetSupportedImageFormats getSupportedImageFormats;
    fn_clGetMemObjectInfo getMemObjectInfo;
    fn_clGetImageInfo getImageInfo;
    fn_clCreateSampler createSampler;
    fn_clRetainSampler retainSampler;
    fn_clReleaseSampler releaseSampler;
    fn_clGetSamplerInfo getSamplerInfo;
    fn_clCreateProgramWithSource createProgramWithSource;
    fn_clCreateProgramWithBinary createProgramWithBinary;
    fn_clRetainProgram retainProgram;
    fn_clReleaseProgram releaseProgram;
    fn_clBuildProgram buildProgram;
    fn_clUnloadCompiler unloadCompiler;
    fn_clGetProgramInfo getProgramInfo;
    fn_clGetProgramBuildInfo getProgramBuildInfo;
    fn_clCreateKernel createKernel;
    fn_clCreateKernelsInProgram createKernelsInProgram;
    fn_clRetainKernel retainKernel;
    fn_clReleaseKernel releaseKernel;
    fn_clSetKernelArg setKernelArg;
    fn_clGetKernelInfo getKernelInfo;
    fn_clGetKernelWorkGroupInfo getKernelWorkGroupInfo;
    fn_clWaitForEvents waitForEvents;
    fn_clGetEventInfo getEventInfo;
    fn_clRetainEvent retainEvent;
    fn_clReleaseEvent releaseEvent;
    fn_clGetEventProfilingInfo getEventProfilingInfo;
    fn_clFlush flush;
    fn_clFinish finish;
    fn_clEnqueueReadBuffer enqueueReadBuffer;
    fn_clEnqueueWriteBuffer enqueueWriteBuffer;
    fn_clEnqueueCopyBuffer enqueueCopyBuffer;
    fn_clEnqueueReadImage enqueueReadImage;
    fn_clEnqueueWriteImage enqueueWriteImage;
    fn_clEnqueueCopyImage enqueueCopyImage;
    fn_clEnqueueCopyImageToBuffer enqueueCopyImageToBuffer;
    fn_clEnqueueCopyBufferToImage enqueueCopyBufferToImage;
    fn_clEnqueueMapBuffer enqueueMapBuffer;
    fn_clEnqueueMapImage enqueueMapImage;
    fn_clEnqueueUnmapMemObject enqueueUnmapMemObject;
    fn_clEnqueueNDRangeKernel enqueueNDRAngeKernel;
    fn_clEnqueueTask enqueueTask;
    fn_clEnqueueNativeKernel enqueueNativeKernel;
    fn_clEnqueueMarker enqueueMarker;
    fn_clEnqueueWaitForEvents enqueueWaitForEvents;
    fn_clEnqueueBarrier enqueueBarrier;
    fn_clGetExtensionFunctionAddress getExtensionFunctionAddress;
} CL_FUNCTIONS;

extern CL_FUNCTIONS cl;

#define clGetPlatformIDs cl.getPlatformIDs
#define clGetPlatformInfo cl.getPlatformInfo
#define clGetDeviceIDs cl.getDeviceIDs
#define clGetDeviceInfo cl.getDeviceInfo
#define clCreateContext cl.createContext
#define clCreateContextFromType cl.createContextFromType
#define clRetainContext cl.retainContext
#define clReleaseContext cl.releaseContext
#define clGetContextInfo cl.getContextInfo
#define clCreateCommandQueue cl.createCommandQueue
#define clRetainCommandQueue cl.retainCommandQueue
#define clReleaseCommandQueue cl.releaseCommandQueue
#define clGetCommandQueueInfo cl.getCommandQueue
#define clCreateBuffer cl.createBuffer
#define clCreateSubBuffer cl.createSubBuffer
#define clCreateImage2D cl.createImage2D
#define clCreateImage3D cl.createImage3D
#define clRetainMemObject cl.retainMemObject
#define clReleaseMemObject cl.releaseMemObject
#define clGetSupportedImageFormats cl.getSupportedImageFormats
#define clGetMemObjectInfo cl.getMemObjectInfo
#define clGetImageInfo cl.getImageInfo
#define clSetMemObjectDestructorCallback cl.setMemObjectDestructorCallback
#define clCreateSampler cl.createSampler
#define clRetainSampler cl.retainSampler
#define clReleaseSampler cl.releaseSampler
#define clGetSamplerInfo cl.getSamplerInfo
#define clCreateProgramWithSource cl.createProgramWithSource
#define clCreateProgramWithBinary cl.createProgramWithBinary
#define clRetainProgram cl.retainProgram
#define clReleaseProgram cl.releaseProgram
#define clBuildProgram cl.buildProgram
#define clUnloadCompiler cl.unloadCompiler
#define clGetProgramInfo cl.getProgramInfo
#define clGetProgramBuildInfo cl.getProgramBuildInfo
#define clCreateKernel cl.createKernel
#define clCreateKernelsInProgram cl.createKernelsInProgram
#define clRetainKernel cl.retainKernel
#define clReleaseKernel cl.releaseKernel
#define clSetKernelArg cl.setKernelArg
#define clGetKernelInfo cl.getKernelInfo
#define clGetKernelWorkGroupInfo cl.getKernelWorkGroupInfo
#define clWaitForEvents cl.waitForEvents
#define clGetEventInfo cl.getEventInfo
#define clCreateUserEvent cl.createUserEvent
#define clRetainEvent cl.retainEvent
#define clReleaseEvent cl.releaseEvent
#define clSetUserEventStatus cl.setUserEventStatus
#define clSetEventCallback cl.setEventCallback
#define clGetEventProfilingInfo cl.getEventProfilingInfo
#define clFlush cl.flush
#define clFinish cl.finish
#define clEnqueueReadBuffer cl.enqueueReadBuffer
#define clEnqueueReadBufferRect cl.enqueueReadBufferRect
#define clEnqueueWriteBuffer cl.enqueueWriteBuffer
#define clEnqueueWriteBufferRect cl.enqueueWriteBufferRect
#define clEnqueueCopyBuffer cl.enqueueCopyBuffer
#define clEnqueueCopyBufferRect cl.enqueueCopyBufferRect
#define clEnqueueReadImage cl.enqueueReadImage
#define clEnqueueWriteImage cl.enqueueWriteImage
#define clEnqueueCopyImage cl.enqueueCopyImage
#define clEnqueueCopyImageToBuffer cl.enqueueCopyImageToBuffer
#define clEnqueueCopyBufferToImage cl.enqueueCopyBufferToImage
#define clEnqueueMapBuffer cl.enqueueMapBuffer
#define clEnqueueMapImage cl.enqueueMapImage
#define clEnqueueUnmapMemObject cl.enqueueUnmapMemObject
#define clEnqueueNDRangeKernel cl.enqueueNDRAngeKernel
#define clEnqueueTask cl.enqueueTask
#define clEnqueueNativeKernel cl.enqueueNativeKernel
#define clEnqueueMarker cl.enqueueMarker
#define clEnqueueWaitForEvents cl.enqueueWaitForEvents
#define clEnqueueBarrier cl.enqueueBarrier
#define clGetExtensionFunctionAddress cl.getExtensionFunctionAddress

#define CL_LOAD_FN(name, ref) \
    ref = dlsym(dll,name); \
    if (ref == NULL){ \
        fprintf(stderr, "Couldn't find %s\n", name); \
        dlclose(dll); \
        return 0; \
    } else { \
        printf("Found CL function %s at %p\n",name,ref); \
    } \



#ifdef	__cplusplus
}
#endif

#endif	/* DYNAMIC_CL_H */

