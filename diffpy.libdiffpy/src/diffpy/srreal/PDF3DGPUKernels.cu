/*************************************************************************************
*
* CUDA kernels for PDF3DGPUCalculator
*
*************************************************************************************/

#include <diffpy/srreal/PDF3DGPUKernels.hpp>

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace diffpy {
namespace srreal {

namespace {

__global__
void accumulate_pairs_kernel(
    double* grid,
    std::size_t grid_size,
    int nbins,
    double dr,
    const PDF3DGpuPair* pairs,
    std::size_t pair_count)
{
    const std::size_t pair_index = blockIdx.x;
    if (pair_index >= pair_count)  return;

    const PDF3DGpuPair p = pairs[pair_index];
    const int nx = p.ix1 - p.ix0 + 1;
    const int ny = p.iy1 - p.iy0 + 1;
    const int nz = p.iz1 - p.iz0 + 1;
    const int local_count = nx * ny * nz;

    for (int local = threadIdx.x; local < local_count; local += blockDim.x)
    {
        const int lx = local % nx;
        const int ly = (local / nx) % ny;
        const int lz = local / (nx * ny);

        const int ix = p.ix0 + lx;
        const int iy = p.iy0 + ly;
        const int iz = p.iz0 + lz;

        const double halfspan = (nbins / 2) * dr;
        const double dx = ix * dr - halfspan - p.rx;
        const double dy = iy * dr - halfspan - p.ry;
        const double dz = iz * dr - halfspan - p.rz;

        const double tx =
            p.sigma_inv[0] * dx + p.sigma_inv[1] * dy + p.sigma_inv[2] * dz;
        const double ty =
            p.sigma_inv[3] * dx + p.sigma_inv[4] * dy + p.sigma_inv[5] * dz;
        const double tz =
            p.sigma_inv[6] * dx + p.sigma_inv[7] * dy + p.sigma_inv[8] * dz;
        const double mahalanobis_sq = dx * tx + dy * ty + dz * tz;
        if (mahalanobis_sq > 16.0)  continue;

        const std::size_t gidx =
            (static_cast<std::size_t>(iz) * nbins + iy) * nbins + ix;
        if (gidx >= grid_size)  continue;

        const double val = p.norm_factor * exp(-0.5 * mahalanobis_sq);
        atomicAdd(&grid[gidx], val);
    }
}

__global__
void copy_centered_grid_kernel(
    cufftDoubleComplex* data,
    const double* grid,
    int nbins,
    int npad)
{
    const std::size_t total =
        static_cast<std::size_t>(nbins) * nbins * nbins;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < total; idx += blockDim.x * gridDim.x)
    {
        const int ix = idx % nbins;
        const int iy = (idx / nbins) % nbins;
        const int iz = idx / (nbins * nbins);
        const int center = nbins / 2;
        const int sx = (ix - center + npad) % npad;
        const int sy = (iy - center + npad) % npad;
        const int sz = (iz - center + npad) % npad;
        const std::size_t dst =
            (static_cast<std::size_t>(sz) * npad + sy) * npad + sx;
        data[dst].x = grid[idx];
        data[dst].y = 0.0;
    }
}

__global__
void apply_qwindow_kernel(
    cufftDoubleComplex* data,
    int npad,
    double qstep,
    double qmin,
    double qmax)
{
    const std::size_t total =
        static_cast<std::size_t>(npad) * npad * npad;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < total; idx += blockDim.x * gridDim.x)
    {
        const int ix = idx % npad;
        const int iy = (idx / npad) % npad;
        const int iz = idx / (npad * npad);
        const int kx = (ix <= npad / 2) ? ix : ix - npad;
        const int ky = (iy <= npad / 2) ? iy : iy - npad;
        const int kz = (iz <= npad / 2) ? iz : iz - npad;
        const double qx = kx * qstep;
        const double qy = ky * qstep;
        const double qz = kz * qstep;
        const double q = sqrt(qx * qx + qy * qy + qz * qz);
        if ((qmax > 0.0 && q > qmax) || (qmin > 0.0 && q < qmin))
        {
            data[idx].x = 0.0;
            data[idx].y = 0.0;
        }
    }
}

__global__
void copy_uncentered_grid_kernel(
    double* grid,
    const cufftDoubleComplex* data,
    int nbins,
    int npad,
    double norm)
{
    const std::size_t total =
        static_cast<std::size_t>(nbins) * nbins * nbins;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < total; idx += blockDim.x * gridDim.x)
    {
        const int ix = idx % nbins;
        const int iy = (idx / nbins) % nbins;
        const int iz = idx / (nbins * nbins);
        const int center = nbins / 2;
        const int sx = (ix - center + npad) % npad;
        const int sy = (iy - center + npad) % npad;
        const int sz = (iz - center + npad) % npad;
        const std::size_t src =
            (static_cast<std::size_t>(sz) * npad + sy) * npad + sx;
        grid[idx] = data[src].x * norm;
    }
}

__global__
void postprocess_kernel(
    double* grid,
    std::size_t grid_size,
    int nbins,
    double dr,
    double rdf_scale,
    double rho0_background,
    double qdamp)
{
    const double halfspan = (nbins / 2) * dr;
    const bool use_qdamp = qdamp > 0.0;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < grid_size; idx += blockDim.x * gridDim.x)
    {
        double val = grid[idx];
        if (rdf_scale != 1.0)  val *= rdf_scale;
        if (rho0_background != 0.0)  val -= rho0_background;
        if (use_qdamp && val != 0.0)
        {
            const int ix = idx % nbins;
            const int iy = (idx / nbins) % nbins;
            const int iz = idx / (nbins * nbins);
            const double x = ix * dr - halfspan;
            const double y = iy * dr - halfspan;
            const double z = iz * dr - halfspan;
            const double r = sqrt(x * x + y * y + z * z);
            val *= exp(-0.5 * (r * qdamp) * (r * qdamp));
        }
        grid[idx] = val;
    }
}

void check_cuda(cudaError_t err, const char* what)
{
    if (err == cudaSuccess)  return;
    throw std::runtime_error(
        std::string(what) + ": " + cudaGetErrorString(err));
}

void check_cufft(cufftResult err, const char* what)
{
    if (err == CUFFT_SUCCESS)  return;
    throw std::runtime_error(
        std::string(what) + ": cuFFT error " + std::to_string(err));
}

}   // namespace

bool pdf3d_cuda_available()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void pdf3d_accumulate_cuda(
    double* grid,
    std::size_t grid_size,
    int nbins,
    double dr,
    const PDF3DGpuPair* pairs,
    std::size_t pair_count)
{
    if (pair_count == 0 || grid_size == 0)  return;

    double* d_grid = 0;
    PDF3DGpuPair* d_pairs = 0;
    const std::size_t grid_bytes = grid_size * sizeof(double);
    const std::size_t pair_bytes = pair_count * sizeof(PDF3DGpuPair);

    check_cuda(cudaMalloc(&d_grid, grid_bytes), "cudaMalloc grid");
    check_cuda(cudaMalloc(&d_pairs, pair_bytes), "cudaMalloc pairs");

    try
    {
        check_cuda(cudaMemcpy(d_grid, grid, grid_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy grid H2D");
        check_cuda(cudaMemcpy(d_pairs, pairs, pair_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy pairs H2D");

        const int threads = 256;
        accumulate_pairs_kernel<<<static_cast<unsigned int>(pair_count), threads>>>(
            d_grid, grid_size, nbins, dr, d_pairs, pair_count);
        check_cuda(cudaGetLastError(), "accumulate_pairs_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "accumulate_pairs_kernel sync");

        check_cuda(cudaMemcpy(grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy grid D2H");
    }
    catch (...)
    {
        cudaFree(d_pairs);
        cudaFree(d_grid);
        throw;
    }

    cudaFree(d_pairs);
    cudaFree(d_grid);
}

void pdf3d_apply_qwindow_cuda(
    double* grid,
    int nbins,
    double dr,
    double qmin,
    double qmax)
{
    if (nbins <= 0 || (qmin <= 0.0 && qmax <= 0.0))  return;

    int npad = 1;
    while (npad < nbins)  npad <<= 1;

    const std::size_t grid_size =
        static_cast<std::size_t>(nbins) * nbins * nbins;
    const std::size_t padded_size =
        static_cast<std::size_t>(npad) * npad * npad;
    const std::size_t grid_bytes = grid_size * sizeof(double);
    const std::size_t data_bytes = padded_size * sizeof(cufftDoubleComplex);

    double* d_grid = 0;
    cufftDoubleComplex* d_data = 0;
    cufftHandle plan = 0;

    check_cuda(cudaMalloc(&d_grid, grid_bytes), "cudaMalloc qwindow grid");
    check_cuda(cudaMalloc(&d_data, data_bytes), "cudaMalloc qwindow data");

    try
    {
        check_cuda(cudaMemcpy(d_grid, grid, grid_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy qwindow grid H2D");
        check_cuda(cudaMemset(d_data, 0, data_bytes), "cudaMemset qwindow data");

        const int threads = 256;
        const int grid_blocks = static_cast<int>(
            std::min<std::size_t>((grid_size + threads - 1) / threads, 65535));
        const int padded_blocks = static_cast<int>(
            std::min<std::size_t>((padded_size + threads - 1) / threads, 65535));

        copy_centered_grid_kernel<<<grid_blocks, threads>>>(
            d_data, d_grid, nbins, npad);
        check_cuda(cudaGetLastError(), "copy_centered_grid_kernel launch");

        check_cufft(cufftPlan3d(&plan, npad, npad, npad, CUFFT_Z2Z),
            "cufftPlan3d");
        check_cufft(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD),
            "cufftExecZ2Z forward");

        const double qstep = 2.0 * 3.14159265358979323846 / (npad * dr);
        apply_qwindow_kernel<<<padded_blocks, threads>>>(
            d_data, npad, qstep, qmin, qmax);
        check_cuda(cudaGetLastError(), "apply_qwindow_kernel launch");

        check_cufft(cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE),
            "cufftExecZ2Z inverse");

        const double norm = 1.0 / static_cast<double>(padded_size);
        copy_uncentered_grid_kernel<<<grid_blocks, threads>>>(
            d_grid, d_data, nbins, npad, norm);
        check_cuda(cudaGetLastError(), "copy_uncentered_grid_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "qwindow sync");

        check_cuda(cudaMemcpy(grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy qwindow grid D2H");
    }
    catch (...)
    {
        if (plan)  cufftDestroy(plan);
        cudaFree(d_data);
        cudaFree(d_grid);
        throw;
    }

    if (plan)  cufftDestroy(plan);
    cudaFree(d_data);
    cudaFree(d_grid);
}

void pdf3d_postprocess_cuda(
    double* grid,
    std::size_t grid_size,
    int nbins,
    double dr,
    double rdf_scale,
    double rho0_background,
    double qdamp)
{
    if (grid_size == 0)  return;
    if (rdf_scale == 1.0 && rho0_background == 0.0 && qdamp <= 0.0)  return;

    double* d_grid = 0;
    const std::size_t grid_bytes = grid_size * sizeof(double);
    check_cuda(cudaMalloc(&d_grid, grid_bytes), "cudaMalloc postprocess grid");

    try
    {
        check_cuda(cudaMemcpy(d_grid, grid, grid_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy postprocess grid H2D");
        const int threads = 256;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((grid_size + threads - 1) / threads, 65535));
        postprocess_kernel<<<blocks, threads>>>(
            d_grid, grid_size, nbins, dr, rdf_scale, rho0_background, qdamp);
        check_cuda(cudaGetLastError(), "postprocess_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "postprocess_kernel sync");
        check_cuda(cudaMemcpy(grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy postprocess grid D2H");
    }
    catch (...)
    {
        cudaFree(d_grid);
        throw;
    }

    cudaFree(d_grid);
}

}   // namespace srreal
}   // namespace diffpy
