/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef BLOCKD_OPENCL_H
#define BLOCKD_OPENCL_H

#ifdef	__cplusplus
extern "C" {
#endif

#include "vp8_opencl.h"
#include "vp8/common/blockd.h"

    extern int vp8_cl_block_finish(BLOCKD *b);
    extern int vp8_cl_block_prep(BLOCKD *b);

#ifdef	__cplusplus
}
#endif

#endif