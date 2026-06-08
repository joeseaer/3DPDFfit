/*************************************************************************************
*
* Implementation of PDF3DGPUCalculator
*
*************************************************************************************/

#include <diffpy/srreal/PDF3DGPUCalculator.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <diffpy/serialization.hpp>
#include <diffpy/srreal/R3linalg.hpp> // For R3::Matrix and R3::Vector, and eigen_solve_3x3
#include <diffpy/srreal/PQEvaluator.hpp>
#include <gsl/gsl_fft_complex.h>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

namespace diffpy {
namespace srreal {

// Static constants
constexpr double PDF3DGPUCalculator::DEFAULT_RMAX_3D;
constexpr double PDF3DGPUCalculator::DEFAULT_GRID_STEP;

// Constructor ---------------------------------------------------------------

PDF3DGPUCalculator::PDF3DGPUCalculator() :
    mdr(DEFAULT_GRID_STEP),
    mnbins(0),
    maccumblocksize(32),
    mapplyrho0background3d(true),
    musecqwindow3d(true),
    mtotaloccupancy(0.0),
    msfaverage(0.0)
{
    this->setRmax(DEFAULT_RMAX_3D);
    this->setRstep(mdr);
    this->setQmax(12.0);
}

// Public Methods ------------------------------------------------------------

void PDF3DGPUCalculator::setGridStep(double dr)
{
    if (dr <= 0) throw std::invalid_argument("Grid step must be positive.");
    if (dr != mdr)
    {
        mdr = dr;
        this->setRstep(dr);
        this->resetValue(); // triggers re-allocation
    }
}

double PDF3DGPUCalculator::getGridStep() const
{
    return mdr;
}

void PDF3DGPUCalculator::setAccumBlockSize(int bs)
{
    if (bs <= 0) throw std::invalid_argument("Accumulation block size must be positive.");
    maccumblocksize = bs;
}

int PDF3DGPUCalculator::getAccumBlockSize() const
{
    return maccumblocksize;
}

void PDF3DGPUCalculator::setApplyRho0Background3D(bool v)
{
    mapplyrho0background3d = v;
}

bool PDF3DGPUCalculator::getApplyRho0Background3D() const
{
    return mapplyrho0background3d;
}

void PDF3DGPUCalculator::setUseCQWindow3D(bool v)
{
    musecqwindow3d = v;
}

bool PDF3DGPUCalculator::getUseCQWindow3D() const
{
    return musecqwindow3d;
}

QuantityType PDF3DGPUCalculator::get3DPDF() const
{
    QuantityType result;
    std::vector<double> grid = mgrid3d;
    if (musecqwindow3d) applyQWindow3D(grid);
    applyPostProcessing3D(grid);

    size_t nonzero_count = 0;
    for (double val : grid) {
        if (val != 0.0) ++nonzero_count;
    }
    result.reserve(nonzero_count * 4);

    for (size_t i = 0; i < grid.size(); ++i)
    {
        if (grid[i] != 0.0)
        {
            double x, y, z;
            indexToCoord(i, x, y, z);

            result.push_back(x);
            result.push_back(y);
            result.push_back(z);
            result.push_back(grid[i]);
        }
    }
    return result;
}

void PDF3DGPUCalculator::exportGrid3DBinary(const std::string& path, bool usefloat32, bool applypost) const
{
    std::vector<double> grid = mgrid3d;
    if (applypost)
    {
        if (musecqwindow3d) applyQWindow3D(grid);
        applyPostProcessing3D(grid);
    }

    std::ofstream ofs(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!ofs) throw std::runtime_error("Failed to open output file: " + path);

    const size_t nxy = static_cast<size_t>(mnbins) * mnbins;
    if (usefloat32)
    {
        std::vector<float> row(nxy);
        for (int iz = 0; iz < mnbins; ++iz)
        {
            const size_t base = static_cast<size_t>(iz) * nxy;
            for (size_t i = 0; i < nxy; ++i) row[i] = static_cast<float>(grid[base + i]);
            ofs.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(nxy * sizeof(float)));
        }
    }
    else
    {
        for (int iz = 0; iz < mnbins; ++iz)
        {
            const size_t base = static_cast<size_t>(iz) * nxy;
            ofs.write(reinterpret_cast<const char*>(&grid[base]), static_cast<std::streamsize>(nxy * sizeof(double)));
        }
    }
    if (!ofs) throw std::runtime_error("Failed while writing output file: " + path);
}

double PDF3DGPUCalculator::computeRho0Background() const
{
    const StructureAdapterPtr& structure = this->getStructure();
    if (!structure)  return 0.0;
    const double partialpdfscale = this->getPartialPDFScale();
    return partialpdfscale * structure->numberDensity();
}

void PDF3DGPUCalculator::applyQWindow3D(std::vector<double>& grid) const
{
    if (grid.empty())  return;

    const double qmin = this->getQmin();
    const double qmax = this->getQmax();
    if (qmin <= 0.0 && qmax <= 0.0)  return;

#ifdef DIFFPY_ENABLE_CUDA
    if (pdf3d_cuda_available())
    {
        pdf3d_apply_qwindow_cuda(grid.data(), mnbins, mdr, qmin, qmax);
        return;
    }
#endif

    const int n = mnbins;
    int npad = 1;
    while (npad < n)  npad <<= 1;

    const size_t n3 = static_cast<size_t>(npad) * npad * npad;
    std::vector<double> data(2 * n3, 0.0);

    const int center = n / 2;
    for (int iz = 0; iz < n; ++iz)
    {
        const int sz = (iz - center + npad) % npad;
        for (int iy = 0; iy < n; ++iy)
        {
            const int sy = (iy - center + npad) % npad;
            const size_t base_src = static_cast<size_t>(iz * n + iy) * n;
            const size_t base_dst = static_cast<size_t>(sz * npad + sy) * npad;
            for (int ix = 0; ix < n; ++ix)
            {
                const int sx = (ix - center + npad) % npad;
                const size_t src = base_src + ix;
                const size_t dst = base_dst + sx;
                data[2 * dst] = grid[src];
            }
        }
    }

    for (int iz = 0; iz < npad; ++iz)
    {
        for (int iy = 0; iy < npad; ++iy)
        {
            double* row = &data[2 * ((iz * npad + iy) * npad)];
            gsl_fft_complex_radix2_forward(row, 1, npad);
        }
    }

    for (int iz = 0; iz < npad; ++iz)
    {
        for (int ix = 0; ix < npad; ++ix)
        {
            double* col = &data[2 * (iz * npad * npad + ix)];
            gsl_fft_complex_radix2_forward(col, npad, npad);
        }
    }

    for (int iy = 0; iy < npad; ++iy)
    {
        for (int ix = 0; ix < npad; ++ix)
        {
            double* line = &data[2 * (iy * npad + ix)];
            gsl_fft_complex_radix2_forward(line, npad * npad, npad);
        }
    }

    const double qstep = (npad > 0) ? (2.0 * M_PI / (npad * mdr)) : 0.0;
    for (int iz = 0; iz < npad; ++iz)
    {
        const int kz = (iz <= npad / 2) ? iz : iz - npad;
        const double qz = kz * qstep;
        for (int iy = 0; iy < npad; ++iy)
        {
            const int ky = (iy <= npad / 2) ? iy : iy - npad;
            const double qy = ky * qstep;
            for (int ix = 0; ix < npad; ++ix)
            {
                const int kx = (ix <= npad / 2) ? ix : ix - npad;
                const double qx = kx * qstep;
                const double q = sqrt(qx * qx + qy * qy + qz * qz);
                if ((qmax > 0.0 && q > qmax) || (qmin > 0.0 && q < qmin))
                {
                    const size_t idx = (static_cast<size_t>(iz) * npad + iy) * npad + ix;
                    data[2 * idx] = 0.0;
                    data[2 * idx + 1] = 0.0;
                }
            }
        }
    }

    for (int iy = 0; iy < npad; ++iy)
    {
        for (int ix = 0; ix < npad; ++ix)
        {
            double* line = &data[2 * (iy * npad + ix)];
            gsl_fft_complex_radix2_inverse(line, npad * npad, npad);
        }
    }

    for (int iz = 0; iz < npad; ++iz)
    {
        for (int ix = 0; ix < npad; ++ix)
        {
            double* col = &data[2 * (iz * npad * npad + ix)];
            gsl_fft_complex_radix2_inverse(col, npad, npad);
        }
    }

    for (int iz = 0; iz < npad; ++iz)
    {
        for (int iy = 0; iy < npad; ++iy)
        {
            double* row = &data[2 * ((iz * npad + iy) * npad)];
            gsl_fft_complex_radix2_inverse(row, 1, npad);
        }
    }

    for (int iz = 0; iz < n; ++iz)
    {
        const int sz = (iz - center + npad) % npad;
        for (int iy = 0; iy < n; ++iy)
        {
            const int sy = (iy - center + npad) % npad;
            const size_t base_src = static_cast<size_t>(sz * npad + sy) * npad;
            const size_t base_dst = static_cast<size_t>(iz * n + iy) * n;
            for (int ix = 0; ix < n; ++ix)
            {
                const int sx = (ix - center + npad) % npad;
                const size_t src = base_src + sx;
                const size_t dst = base_dst + ix;
                grid[dst] = data[2 * src];
            }
        }
    }
}

void PDF3DGPUCalculator::applyPostProcessing3D(std::vector<double>& grid) const
{
    const double rdf_scale = (mtotaloccupancy * msfaverage == 0.0) ? 0.0 :
        1.0 / (mtotaloccupancy * msfaverage * msfaverage);
    const double rho0_bg = mapplyrho0background3d ?
        computeRho0Background() : 0.0;

    double qdamp = 0.0;
    try
    {
        qdamp = this->getEnvelopeByType("qresolution")->getDoubleAttr("qdamp");
    }
    catch (...)
    {
        qdamp = 0.0;
    }

#ifdef DIFFPY_ENABLE_CUDA
    if (pdf3d_cuda_available())
    {
        pdf3d_postprocess_cuda(
            grid.data(), grid.size(), mnbins, mdr,
            rdf_scale, rho0_bg, qdamp);
        return;
    }
#endif

    if (rdf_scale != 1.0)
    {
        for (double& val : grid) val *= rdf_scale;
    }

    if (rho0_bg != 0.0)
    {
        for (double& val : grid) val -= rho0_bg;
    }

    if (qdamp > 0.0)
    {
        for (size_t i = 0; i < grid.size(); ++i)
        {
            if (grid[i] == 0.0) continue;
            double x, y, z;
            indexToCoord(i, x, y, z);
            const double r = sqrt(x * x + y * y + z * z);
            grid[i] *= exp(-0.5 * (r * qdamp) * (r * qdamp));
        }
    }
}

// Protected Methods ---------------------------------------------------------

void PDF3DGPUCalculator::resetValue()
{
    const StructureAdapterPtr& structure = this->getStructure();
    if (!structure)
    {
        msfCache.clear();
        mtotaloccupancy = 0.0;
        msfaverage = 0.0;
    }
    else
    {
        int nsite = this->countSites();
        msfCache.resize(nsite);
        const auto& sftable = this->getScatteringFactorTable();

        mtotaloccupancy = structure->totalOccupancy();
        double totsf = 0.0;
        for (int i = 0; i < nsite; ++i)
        {
            std::string atom_type = structure->siteAtomType(i);
            const double sf = sftable->lookup(atom_type);
            const double occ = structure->siteOccupancy(i);
            const double mult = structure->siteMultiplicity(i);
            msfCache[i] = sf * occ;
            totsf += msfCache[i] * mult;
        }
        msfaverage = (mtotaloccupancy == 0.0) ? 0.0 : (totsf / mtotaloccupancy);
    }

    // Ensure odd number of bins so origin is centered.
    mnbins = static_cast<int>(2 * ceil(this->getRmax() / mdr)) + 1;
    size_t total = static_cast<size_t>(mnbins) * mnbins * mnbins;
    mgrid3d.assign(total, 0.0);
    mpendinggpupairs.clear();
    
    PDFCalculator::resetValue();
    if (mevaluator) mevaluator->setFlag(USEFULLSUM, true);
}

void PDF3DGPUCalculator::addPairContribution(const BaseBondGenerator& bnds, int summationscale)
{
    if (bnds.distance() == 0.0) return;

    int i0 = bnds.site0();
    int i1 = bnds.site1();
    
    if (i0 >= static_cast<int>(msfCache.size()) || i1 >= static_cast<int>(msfCache.size()))
        return;

    double sfprod = msfCache[i0] * msfCache[i1] *
        bnds.multiplicity() * static_cast<double>(summationscale);

    const R3::Vector& rvec = bnds.r01();
    const R3::Matrix& U_i = bnds.Ucartesian0();
    const R3::Matrix& U_j = bnds.Ucartesian1();

    addAnisotropicGaussianToGrid(rvec, U_i, U_j, sfprod);
}

void PDF3DGPUCalculator::finishValue()
{
    flushPendingGpuPairs();
}

void PDF3DGPUCalculator::addAnisotropicGaussianToGrid(const R3::Vector& r_ij, const R3::Matrix& U_i, const R3::Matrix& U_j, double sfprod)
{
    // 1. Compute total covariance matrix
    R3::Matrix Sigma = U_i + U_j;

    // Eigenvalue decomposition
    R3::Vector eigenvalues;
    R3::Matrix eigenvectors;
    
    R3::eigen_solve_3x3(Sigma, eigenvalues, eigenvectors);

    // Check for positive definiteness (eigenvalues are sorted ascending)
    if (eigenvalues[0] <= 1e-8) {
        return; 
    }

    // 3. Invert the covariance matrix
    R3::Matrix Sigma_inv = R3::inverse(Sigma);
    double det_Sigma = R3::determinant(Sigma);

    // 4. Normalization factor
    const double two_pi = 2.0 * M_PI;
    double norm_factor = sfprod / (pow(two_pi, 1.5) * sqrt(det_Sigma));

    // 5. Determine sampling bounding box
    double max_sigma = 0.0;
    for(int k=0; k<3; ++k) {
        max_sigma = std::max(max_sigma, sqrt(eigenvalues[k]));
    }
    
    // 4.0 sigma cutoff
    double cutoff_radius = 4.0 * max_sigma;
    
    // Safety clamps
    if (cutoff_radius < mdr) cutoff_radius = mdr; 
    if (cutoff_radius > 10.0) cutoff_radius = 10.0; 

    // 6. Iterate over the local grid indices
    double halfspan = (mnbins / 2) * mdr;
    
    auto get_index_range = [&](double center_val) -> std::pair<int, int> {
        double min_val = center_val - cutoff_radius;
        double max_val = center_val + cutoff_radius;
        
        // Node-centered grid: points at k * mdr
        int start = static_cast<int>(ceil((min_val + halfspan) / mdr));
        int end   = static_cast<int>(floor((max_val + halfspan) / mdr));
        
        start = std::max(0, start);
        end   = std::min(mnbins - 1, end);
        
        return {start, end};
    };

    std::pair<int, int> xr = get_index_range(r_ij[0]);
    std::pair<int, int> yr = get_index_range(r_ij[1]);
    std::pair<int, int> zr = get_index_range(r_ij[2]);

    double cutoff_sq = 16.0;

#ifdef DIFFPY_ENABLE_CUDA
    if (pdf3d_cuda_available())
    {
        PDF3DGpuPair pair;
        pair.rx = r_ij[0];
        pair.ry = r_ij[1];
        pair.rz = r_ij[2];
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                pair.sigma_inv[3 * row + col] = Sigma_inv(row, col);
            }
        }
        pair.norm_factor = norm_factor;
        pair.ix0 = xr.first;
        pair.ix1 = xr.second;
        pair.iy0 = yr.first;
        pair.iy1 = yr.second;
        pair.iz0 = zr.first;
        pair.iz1 = zr.second;
        mpendinggpupairs.push_back(pair);
        return;
    }
#endif

    const int bs = std::max(1, maccumblocksize);
    const int nbz = (zr.second - zr.first + bs) / bs;
    const int nby = (yr.second - yr.first + bs) / bs;
    const int nbx = (xr.second - xr.first + bs) / bs;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<double> local;

#ifdef _OPENMP
#pragma omp for collapse(3) schedule(dynamic, 1)
#endif
        for (int tbz = 0; tbz < nbz; ++tbz)
        {
            for (int tby = 0; tby < nby; ++tby)
            {
                for (int tbx = 0; tbx < nbx; ++tbx)
                {
                    const int bz = zr.first + tbz * bs;
                    const int by = yr.first + tby * bs;
                    const int bx = xr.first + tbx * bs;

                    const int izhi = std::min(zr.second, bz + bs - 1);
                    const int iyhi = std::min(yr.second, by + bs - 1);
                    const int ixhi = std::min(xr.second, bx + bs - 1);

                    const int tz = izhi - bz + 1;
                    const int ty = iyhi - by + 1;
                    const int tx = ixhi - bx + 1;
                    const size_t nloc = static_cast<size_t>(tz) * ty * tx;
                    local.assign(nloc, 0.0);

                    bool any = false;
                    for (int iz = bz; iz <= izhi; ++iz)
                    {
                        const double z_grid = iz * mdr - halfspan;
                        const double dz = z_grid - r_ij[2];
                        for (int iy = by; iy <= iyhi; ++iy)
                        {
                            const double y_grid = iy * mdr - halfspan;
                            const double dy = y_grid - r_ij[1];
                            for (int ix = bx; ix <= ixhi; ++ix)
                            {
                                const double x_grid = ix * mdr - halfspan;
                                const double dx = x_grid - r_ij[0];
                                R3::Vector delta(dx, dy, dz);
                                R3::Vector tmp = R3::mxvecproduct(Sigma_inv, delta);
                                const double mahalanobis_sq = R3::dot(delta, tmp);
                                if (mahalanobis_sq > cutoff_sq) continue;

                                const double val = norm_factor * exp(-0.5 * mahalanobis_sq);
                                const int lz = iz - bz;
                                const int ly = iy - by;
                                const int lx = ix - bx;
                                const size_t lidx = (static_cast<size_t>(lz) * ty + ly) * tx + lx;
                                local[lidx] += val;
                                any = true;
                            }
                        }
                    }

                    if (!any) continue;

#ifdef _OPENMP
#pragma omp critical(pdf3d_tile_reduce)
#endif
                    {
                        for (int lz = 0; lz < tz; ++lz)
                        {
                            const int iz = bz + lz;
                            for (int ly = 0; ly < ty; ++ly)
                            {
                                const int iy = by + ly;
                                for (int lx = 0; lx < tx; ++lx)
                                {
                                    const size_t lidx = (static_cast<size_t>(lz) * ty + ly) * tx + lx;
                                    const double v = local[lidx];
                                    if (v == 0.0) continue;
                                    const int ix = bx + lx;
                                    const size_t gidx = (static_cast<size_t>(iz) * mnbins + iy) * mnbins + ix;
                                    mgrid3d[gidx] += v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Private Helpers -----------------------------------------------------------

void PDF3DGPUCalculator::flushPendingGpuPairs()
{
    if (mpendinggpupairs.empty())  return;

#ifdef DIFFPY_ENABLE_CUDA
    pdf3d_accumulate_cuda(
        mgrid3d.data(),
        mgrid3d.size(),
        mnbins,
        mdr,
        mpendinggpupairs.data(),
        mpendinggpupairs.size());
    mpendinggpupairs.clear();
#else
    throw std::logic_error("PDF3DGPUCalculator has pending CUDA work, "
        "but this build was not compiled with DIFFPY_ENABLE_CUDA.");
#endif
}

size_t PDF3DGPUCalculator::coordToIndex(double x, double y, double z) const
{
    double halfspan = (mnbins / 2) * mdr;
    if (fabs(x) > halfspan || fabs(y) > halfspan || fabs(z) > halfspan)
        return mgrid3d.size(); // out of bounds

    int ix = static_cast<int>(floor((x + halfspan) / mdr));
    int iy = static_cast<int>(floor((y + halfspan) / mdr));
    int iz = static_cast<int>(floor((z + halfspan) / mdr));

    ix = std::max(0, std::min(ix, mnbins - 1));
    iy = std::max(0, std::min(iy, mnbins - 1));
    iz = std::max(0, std::min(iz, mnbins - 1));

    return static_cast<size_t>((iz * mnbins + iy) * mnbins + ix);
}

void PDF3DGPUCalculator::indexToCoord(size_t idx, double& x, double& y, double& z) const
{
    size_t iz = idx / (static_cast<size_t>(mnbins) * mnbins);
    size_t rem = idx % (static_cast<size_t>(mnbins) * mnbins);
    size_t iy = rem / mnbins;
    size_t ix = rem % mnbins;

    double halfspan = (mnbins / 2) * mdr;
    x = ix * mdr - halfspan;
    y = iy * mdr - halfspan;
    z = iz * mdr - halfspan;
}

}   // namespace srreal
}   // namespace diffpy

#include <diffpy/serialization.ipp>
DIFFPY_INSTANTIATE_SERIALIZATION(diffpy::srreal::PDF3DGPUCalculator)
