#ifndef FIDESLIB_HIP_COMPAT_CUH
#define FIDESLIB_HIP_COMPAT_CUH
// nvcc-only kernel-parameter attributes with no HIP equivalent.
// Scalar by-value params: dropping the attribute is functionally equivalent on AMD.
#if defined(__HIP__) || defined(__HIPCC__)
#ifndef __grid_constant__
#define __grid_constant__
#endif
#endif
#endif
