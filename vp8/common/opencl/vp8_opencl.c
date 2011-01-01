/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdio.h>
#include "vp8_opencl.h"

int cl_initialized = CL_NOT_INITIALIZED;
VP8_COMMON_CL cl_data;


/**
 *
 */
void cl_destroy() {

    //Wait on any pending operations to complete... frees up all of our pointers
    clFinish(cl_data.commands);

    if (cl_data.filterData){
        clReleaseMemObject(cl_data.filterData);
        cl_data.filterData = NULL;
    }

    if (cl_data.srcData){
        clReleaseMemObject(cl_data.srcData);
        cl_data.srcData = NULL;
        cl_data.srcAlloc = 0;
    }

    if (cl_data.destData){
        clReleaseMemObject(cl_data.destData);
        cl_data.destData = NULL;
        cl_data.destAlloc = 0;
    }

    if (cl_data.intData){
        clReleaseMemObject(cl_data.intData);
        cl_data.intData = NULL;
        cl_data.intAlloc = 0;
        cl_data.intSize = 0;
    }

    //Release the objects that we've allocated on the GPU
    if (cl_data.program)
        clReleaseProgram(cl_data.program);
    if (cl_data.filter_block2d_first_pass_kernel)
        clReleaseKernel(cl_data.filter_block2d_first_pass_kernel);
    if (cl_data.filter_block2d_second_pass_kernel)
        clReleaseKernel(cl_data.filter_block2d_second_pass_kernel);
    if (cl_data.commands)
        clReleaseCommandQueue(cl_data.commands);
    if (cl_data.context)
        clReleaseContext(cl_data.context);

    cl_data.program = NULL;
    cl_data.filter_block2d_first_pass_kernel = NULL;
    cl_data.filter_block2d_second_pass_kernel = NULL;
    cl_data.commands = NULL;
    cl_data.context = NULL;

    cl_initialized = CL_NOT_INITIALIZED;

    return;
}

int cl_init(){
    int err;
    cl_platform_id platform_ids[MAX_NUM_PLATFORMS];
    cl_uint num_found;

    //Don't allow multiple CL contexts..
    if (cl_initialized != CL_NOT_INITIALIZED)
        return cl_initialized;

    // Connect to a compute device
    err = clGetPlatformIDs(MAX_NUM_PLATFORMS, platform_ids, &num_found);

    if (err != CL_SUCCESS) {
        printf("Couldn't query platform IDs\n");
        return CL_TRIED_BUT_FAILED;
    }
    if (num_found == 0) {
        printf("No platforms found\n");
        return CL_TRIED_BUT_FAILED;
    }
    //printf("Found %d platforms\n", num_found);

    //Favor the GPU, but fall back to any other available device if necessary
    err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &cl_data.device_id, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, 1, &cl_data.device_id, NULL);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to create a device group!\n");
            return CL_TRIED_BUT_FAILED;
        }
    }

    // Create the compute context
    cl_data.context = clCreateContext(0, 1, &cl_data.device_id, NULL, NULL, &err);
    if (!cl_data.context) {
        printf("Error: Failed to create a compute context!\n");
        return CL_TRIED_BUT_FAILED;
    }

    // Create a command queue
    cl_data.commands = clCreateCommandQueue(cl_data.context, cl_data.device_id, 0, &err);
    if (!cl_data.commands || err != CL_SUCCESS) {
        printf("Error: Failed to create a command queue!\n");
        return CL_TRIED_BUT_FAILED;
    }

    return CL_SUCCESS;
}

char *cl_read_file(const char* file_name){
    long pos;
    char *bytes;
    size_t amt_read;

    FILE *f = fopen(file_name, "rb");
    if (f == NULL)
        return NULL;

    fseek(f, 0, SEEK_END);
    pos = ftell(f);
    fseek(f, 0, SEEK_SET);

    bytes = malloc(pos);
    if (bytes == NULL){
        fclose(f);
        return NULL;
    }

    amt_read = fread(bytes, pos, 1, f);
    if (amt_read != 1){
        free(bytes);
        fclose(f);
        return NULL;
    }

    fclose(f);

    return bytes;
}