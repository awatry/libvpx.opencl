#include "vpx_ports/config.h"

#include "opencl/vp8_opencl.h"
#include "opencl/vp8_decode_cl.h"

#include <stdio.h>

extern int cl_init_dequant();
extern int cl_destroy_dequant();

int cl_decode_destroy(){
    int err;

    err = cl_destroy_dequant();
    
    return CL_SUCCESS;
}

int cl_decode_init()
{
    int err;
    printf("Initializing opencl decoder-specific programs/kernels");

    //Initialize programs to null value
    //Enables detection of if they've been initialized as well.
    cl_data.dequant_program = NULL;

    err = cl_init_dequant();
    if (err != CL_SUCCESS)
        return err;

    printf(" .. done\n");

    return CL_SUCCESS;
}
