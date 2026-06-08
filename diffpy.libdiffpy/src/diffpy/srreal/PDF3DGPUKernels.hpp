/*************************************************************************************
*
* CUDA backend declarations for PDF3DGPUCalculator
*
*************************************************************************************/

#ifndef PDF3DGPUKERNELS_HPP_INCLUDED
#define PDF3DGPUKERNELS_HPP_INCLUDED

#include <cstddef>

namespace diffpy {
namespace srreal {

struct PDF3DGpuPair
{
    double rx, ry, rz;
    double sigma_inv[9];
    double norm_factor;
    int ix0, ix1;
    int iy0, iy1;
    int iz0, iz1;
};

bool pdf3d_cuda_available();

void pdf3d_accumulate_cuda(
    double* grid,
    std::size_t grid_size,
    int nbins,
    double dr,
    const PDF3DGpuPair* pairs,
    std::size_t pair_count);

void pdf3d_apply_qwindow_cuda(
    double* grid,
    int nbins,
    double dr,
    double qmin,
    double qmax);

void pdf3d_postprocess_cuda(
    double* grid,
    std::size_t grid_size,
    int nbins,
    double dr,
    double rdf_scale,
    double rho0_background,
    double qdamp);

}   // namespace srreal
}   // namespace diffpy

#endif  // PDF3DGPUKERNELS_HPP_INCLUDED
