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
#include <string.h>
#include <stdlib.h>
#include "vp8_opencl.h"

int cl_initialized = CL_NOT_INITIALIZED;
VP8_COMMON_CL cl_data;

//Initialization functions for various CL programs.
extern int cl_init_filter();
extern int cl_init_idct();

/**
 *
 */
void cl_destroy(int new_status) {

    //Wait on any pending operations to complete... frees up all of our pointers
    clFinish(cl_data.commands);

    if (cl_data.srcData) {
        clReleaseMemObject(cl_data.srcData);
        cl_data.srcData = NULL;
        cl_data.srcAlloc = 0;
    }

    if (cl_data.destData) {
        clReleaseMemObject(cl_data.destData);
        cl_data.destData = NULL;
        cl_data.destAlloc = 0;
    }

    if (cl_data.intData) {
        clReleaseMemObject(cl_data.intData);
        cl_data.intData = NULL;
        cl_data.destAlloc = 0;
    }

    //Release the objects that we've allocated on the GPU
    if (cl_data.filter_program)
        clReleaseProgram(cl_data.filter_program);

    if (cl_data.idct_program)
        clReleaseProgram(cl_data.idct_program);

    CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_block_variation_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict8x8_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict8x4_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_sixtap_predict16x16_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict4x4_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict8x4_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict8x8_kernel);
    CL_RELEASE_KERNEL(cl_data.vp8_bilinear_predict16x16_kernel);

    //Older kernels that probably aren't used anymore... remove eventually.
    if (cl_data.filter_block2d_first_pass_kernel)
        clReleaseKernel(cl_data.filter_block2d_first_pass_kernel);
    if (cl_data.filter_block2d_second_pass_kernel)
        clReleaseKernel(cl_data.filter_block2d_second_pass_kernel);
    cl_data.filter_block2d_first_pass_kernel = NULL;
    cl_data.filter_block2d_second_pass_kernel = NULL;


    if (cl_data.commands)
        clReleaseCommandQueue(cl_data.commands);
    if (cl_data.context)
        clReleaseContext(cl_data.context);

    cl_data.filter_program = NULL;
    cl_data.idct_program = NULL;

    cl_data.commands = NULL;
    cl_data.context = NULL;

    cl_initialized = new_status;

    return;
}

int cl_init() {
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

    //Initialize other memory objects to null pointers
    cl_data.srcData = NULL;
    cl_data.srcAlloc = 0;
    cl_data.destData = NULL;
    cl_data.destAlloc = 0;
    cl_data.intData = NULL;
    cl_data.intAlloc = 0;

    //Initialize programs to null value
    //Enables detection of if they've been initialized as well.
    cl_data.filter_program = NULL;
    cl_data.idct_program = NULL;

    err = cl_init_filter();
    if (err != CL_SUCCESS)
        return err;

    err = cl_init_idct();
    if (err != CL_SUCCESS)
        return err;

    return CL_SUCCESS;
}

char *cl_read_file(const char* file_name) {
    long pos;
    char *fullpath, *bak;
    char *bytes;
    size_t amt_read;

    FILE *f;

    f = fopen(file_name, "rb");
    //Disable until this no longer crashes on free()
    if (0 && f == NULL) {

        bak = fullpath = malloc(strlen(vpx_codec_lib_dir() + strlen(file_name) + 2));
        if (fullpath == NULL) {
            return NULL;
        }

        fullpath = strcpy(fullpath, vpx_codec_lib_dir());
        if (fullpath == NULL) {
            free(bak);
            return NULL;
        }

        fullpath = strcat(fullpath, "/");
        if (fullpath == NULL) {
            free(bak);
            return NULL;
        }

        fullpath = strcat(fullpath, file_name);
        if (fullpath == NULL) {
            free(bak);
            return NULL;
        }

        f = fopen(fullpath, "rb");
        if (f == NULL) {
            printf("Couldn't find CL source at %s or %s\n", file_name, fullpath);
            free(fullpath);
            return NULL;
        }

        printf("Found cl source at %s\n", fullpath);
        free(fullpath);
    }

    fseek(f, 0, SEEK_END);
    pos = ftell(f);
    fseek(f, 0, SEEK_SET);
    bytes = malloc(pos+1);

    if (bytes == NULL) {
        fclose(f);
        return NULL;
    }

    amt_read = fread(bytes, pos, 1, f);
    if (amt_read != 1) {
        free(bytes);
        fclose(f);
        return NULL;
    }

    bytes[pos] = '\0'; //null terminate the source string
    fclose(f);


    return bytes;
}

int cl_load_program(cl_program *prog_ref, const char *file_name, const char *opts) {

    int err;
    char *buffer;
    size_t len;
    char *kernel_src = cl_read_file(file_name);
    
    *prog_ref = NULL;
    if (kernel_src != NULL) {
        *prog_ref = clCreateProgramWithSource(cl_data.context, 1, &kernel_src, NULL, &err);
        free(kernel_src);
    } else {
        cl_destroy(CL_TRIED_BUT_FAILED);
        printf("Couldn't find OpenCL source files. \nUsing software path.\n");
        return CL_TRIED_BUT_FAILED;
    }

    if (*prog_ref == NULL) {
        printf("Error: Couldn't create program\n");
        return CL_TRIED_BUT_FAILED;
    }

    if (err != CL_SUCCESS) {
        printf("Error creating program: %d\n", err);
    }

    /* Build the program executable */
    err = clBuildProgram(*prog_ref, 0, NULL, opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to build program executable!\n");
        err = clGetProgramBuildInfo(*prog_ref, cl_data.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        printf("Got log size\n");
        if (err != CL_SUCCESS) {
            printf("Error: Could not get length of CL build log\n");
            return CL_TRIED_BUT_FAILED;
        }
        buffer = (char*) malloc(len);
        if (buffer == NULL) {
            printf("Error: Couldn't allocate compile output buffer memory\n");
            return CL_TRIED_BUT_FAILED;
        }
        printf("Fetching build log\n");
        err = clGetProgramBuildInfo(*prog_ref, cl_data.device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("Got build log\n");
        if (err != CL_SUCCESS) {
            printf("Error: Could not get CL build log\n");
        } else {
            printf("didn't crash in clGetProgramBuildInfo\n");
            printf("Compile output: %s\n", buffer);
        }
        free(buffer);
        return CL_TRIED_BUT_FAILED;
    }
    return CL_SUCCESS;
}
