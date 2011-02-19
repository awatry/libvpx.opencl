/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "../../decoder/onyxd_int.h"
#include "../../../vpx_ports/config.h"
#include "../../common/idct.h"
#include "blockd_cl.h"
#include "../../decoder/opencl/dequantize_cl.h"


int vp8_cl_mb_prep(MACROBLOCKD *x, int flags){
        int err;

    if (cl_initialized != CL_SUCCESS){
        return cl_initialized;
    }

    //Copy all blockd.cl_*_mem objects
    if (flags & DIFF)
        CL_SET_BUF(x->cl_commands, x->cl_diff_mem, sizeof(cl_short)*400, x->diff,
            return err
        );

    if (flags & PREDICTOR)
        CL_SET_BUF(x->cl_commands, x->cl_predictor_mem, sizeof(cl_uchar)*384, x->predictor,
            return err
        );

    if (flags & QCOEFF)
        CL_SET_BUF(x->cl_commands, x->cl_qcoeff_mem, sizeof(cl_short)*400, x->qcoeff,
            return err
        );

    if (flags & DQCOEFF)
        CL_SET_BUF(x->cl_commands, x->cl_dqcoeff_mem, sizeof(cl_short)*400, x->dqcoeff,
            return err
        );

    if (flags & EOBS)
        CL_SET_BUF(x->cl_commands, x->cl_eobs_mem, sizeof(cl_char)*25, x->eobs,
            return err
        );

    if (flags & PRE_BUF){
        CL_SET_BUF(x->cl_commands, x->pre.buffer_mem, x->pre.buffer_size, x->pre.buffer_alloc,
            return err
        );
    }

    if (flags & DST_BUF){
        CL_SET_BUF(x->cl_commands, x->dst.buffer_mem, x->dst.buffer_size, x->dst.buffer_alloc,
            return err
        );
    }


    return CL_SUCCESS;
}

int vp8_cl_mb_finish(MACROBLOCKD *x, int flags){
    int err;

    if (cl_initialized != CL_SUCCESS){
        return cl_initialized;
    }

    if (flags & DIFF){
        err = clEnqueueReadBuffer(x->cl_commands, x->cl_diff_mem, CL_FALSE, 0, sizeof(cl_short)*400, x->diff, 0, NULL, NULL);
        CL_CHECK_SUCCESS( x->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
        );
    }

    if (flags & PREDICTOR){
    err = clEnqueueReadBuffer(x->cl_commands, x->cl_predictor_mem, CL_FALSE, 0, sizeof(cl_uchar)*384, x->predictor, 0, NULL, NULL);
    CL_CHECK_SUCCESS( x->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
    );
    }

    if (flags & QCOEFF){
    err = clEnqueueReadBuffer(x->cl_commands, x->cl_qcoeff_mem, CL_FALSE, 0, sizeof(cl_short)*400, x->qcoeff, 0, NULL, NULL);
    CL_CHECK_SUCCESS( x->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
    );
    }

    if (flags & DQCOEFF){
    err = clEnqueueReadBuffer(x->cl_commands, x->cl_dqcoeff_mem, CL_FALSE, 0, sizeof(cl_short)*400, x->dqcoeff, 0, NULL, NULL);
    CL_CHECK_SUCCESS( x->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
    );
    }

    if (flags & EOBS){
        err = clEnqueueReadBuffer(x->cl_commands, x->cl_eobs_mem, CL_FALSE, 0, sizeof(cl_char)*25, x->eobs, 0, NULL, NULL);
        CL_CHECK_SUCCESS( x->cl_commands, err != CL_SUCCESS,
          "Error: Failed to read from GPU!\n",
            , err
        );
    }

    if (flags & PRE_BUF){
        err = clEnqueueReadBuffer(x->cl_commands, x->pre.buffer_mem, CL_FALSE, 
                0, x->pre.buffer_size, x->pre.buffer_alloc, 0, NULL, NULL);
        CL_CHECK_SUCCESS( x->cl_commands, err != CL_SUCCESS,
          "Error: Failed to read from GPU!\n",
            , err
        );
    }

    if (flags & DST_BUF){
        err = clEnqueueReadBuffer(x->cl_commands, x->dst.buffer_mem, CL_FALSE,
                0, x->dst.buffer_size, x->dst.buffer_alloc, 0, NULL, NULL);
        CL_CHECK_SUCCESS( x->cl_commands, err != CL_SUCCESS,
          "Error: Failed to read from GPU!\n",
            , err
        );
    }


    return CL_SUCCESS;
}

int vp8_cl_block_prep(BLOCKD *b, int flags){
    int err;

    if (cl_initialized != CL_SUCCESS){
        return cl_initialized;
    }

    //Copy all blockd.cl_*_mem objects
    if (flags & DIFF)
        CL_SET_BUF(b->cl_commands, b->cl_diff_mem, sizeof(cl_short)*400, b->diff_base,
            return err
        );

    if (flags & PREDICTOR)
        CL_SET_BUF(b->cl_commands, b->cl_predictor_mem, sizeof(cl_uchar)*384, b->predictor_base,
            return err
        );

    if (flags & QCOEFF)
        CL_SET_BUF(b->cl_commands, b->cl_qcoeff_mem, sizeof(cl_short)*400, b->qcoeff_base,
            return err
        );

    if (flags & DQCOEFF)
        CL_SET_BUF(b->cl_commands, b->cl_dqcoeff_mem, sizeof(cl_short)*400, b->dqcoeff_base,
            return err
        );

    if (flags & EOBS)
        CL_SET_BUF(b->cl_commands, b->cl_eobs_mem, sizeof(cl_char)*25, b->eobs_base,
            return err
        );

    if (flags & DEQUANT)
        CL_SET_BUF(b->cl_commands, b->cl_dequant_mem, sizeof(cl_short)*16 ,b->dequant,
            return err
        );

    return CL_SUCCESS;
}

int vp8_cl_block_finish(BLOCKD *b, int flags){
    int err;

    if (cl_initialized != CL_SUCCESS){
        return cl_initialized;
    }

    if (flags & DIFF){
        err = clEnqueueReadBuffer(b->cl_commands, b->cl_diff_mem, CL_FALSE, 0, sizeof(cl_short)*400, b->diff_base, 0, NULL, NULL);
        CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
        );
    }

    if (flags & PREDICTOR){
    err = clEnqueueReadBuffer(b->cl_commands, b->cl_predictor_mem, CL_FALSE, 0, sizeof(cl_uchar)*384, b->predictor_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
    );
    }

    if (flags & QCOEFF){
    err = clEnqueueReadBuffer(b->cl_commands, b->cl_qcoeff_mem, CL_FALSE, 0, sizeof(cl_short)*400, b->qcoeff_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
    );
    }

    if (flags & DQCOEFF){
    err = clEnqueueReadBuffer(b->cl_commands, b->cl_dqcoeff_mem, CL_FALSE, 0, sizeof(cl_short)*400, b->dqcoeff_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
    );
    }

    if (flags & EOBS){
    err = clEnqueueReadBuffer(b->cl_commands, b->cl_eobs_mem, CL_FALSE, 0, sizeof(cl_char)*25, b->eobs_base, 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
    );
    }

    if (flags & DEQUANT){
    err = clEnqueueReadBuffer(b->cl_commands, b->cl_dequant_mem, CL_FALSE, 0, sizeof(cl_short)*16 ,b->dequant, 0, NULL, NULL);
    CL_CHECK_SUCCESS( b->cl_commands, err != CL_SUCCESS,
        "Error: Failed to read from GPU!\n",
            , err
    );
    }

    return CL_SUCCESS;
}