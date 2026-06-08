/*************************************************************************************
*
* Implementation of PDF3DCPUCalculator
*
*************************************************************************************/

#include <diffpy/srreal/PDF3DCPUCalculator.hpp>
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
constexpr double PDF3DCPUCalculator::DEFAULT_RMAX_3D;
constexpr double PDF3DCPUCalculator::DEFAULT_GRID_STEP;

// Constructor ---------------------------------------------------------------

PDF3DCPUCalculator::PDF3DCPUCalculator() :
    mdr(DEFAULT_GRID_STEP),
    mnbins(0),
    maccumblocksize(32),
    mcpuworkfactor(1),
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

void PDF3DCPUCalculator::setGridStep(double dr)
{
    if (dr <= 0) throw std::invalid_argument("Grid step must be positive.");
    if (dr != mdr)
    {
        mdr = dr;
        this->setRstep(dr);
        this->resetValue(); // triggers re-allocation
    }
}

double PDF3DCPUCalculator::getGridStep() const
{
    return mdr;
}

void PDF3DCPUCalculator::setAccumBlockSize(int bs)
{
    if (bs <= 0) throw std::invalid_argument("Accumulation block size must be positive.");
    maccumblocksize = bs;
}

int PDF3DCPUCalculator::getAccumBlockSize() const
{
    return maccumblocksize;
}

void PDF3DCPUCalculator::setCPUWorkFactor(int factor)
{
    if (factor <= 0) throw std::invalid_argument("CPU work factor must be positive.");
    mcpuworkfactor = factor;
}

int PDF3DCPUCalculator::getCPUWorkFactor() const
{
    return mcpuworkfactor;
}

void PDF3DCPUCalculator::setApplyRho0Background3D(bool v)
{
    mapplyrho0background3d = v;
}

bool PDF3DCPUCalculator::getApplyRho0Background3D() const
{
    return mapplyrho0background3d;
}

void PDF3DCPUCalculator::setUseCQWindow3D(bool v)
{
    musecqwindow3d = v;
}

bool PDF3DCPUCalculator::getUseCQWindow3D() const
{
    return musecqwindow3d;
}

QuantityType PDF3DCPUCalculator::get3DPDF() const
{
    QuantityType result;
    std::vector<double> grid = mgrid3d;
    if (musecqwindow3d) applyQWindow3D(grid);

    const double rdf_scale = (mtotaloccupancy * msfaverage == 0.0) ? 0.0 :
        1.0 / (mtotaloccupancy * msfaverage * msfaverage);
    if (rdf_scale != 1.0)
    {
        for (double& val : grid) val *= rdf_scale;
    }

    if (mapplyrho0background3d)
    {
        const double rho0_bg = computeRho0Background();
        if (rho0_bg != 0.0)
        {
            for (double& val : grid) val -= rho0_bg;
        }
    }

    double qdamp = 0.0;
    try
    {
        qdamp = this->getEnvelopeByType("qresolution")->getDoubleAttr("qdamp");
    }
    catch (...)
    {
        qdamp = 0.0;
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

void PDF3DCPUCalculator::exportGrid3DBinary(const std::string& path, bool usefloat32, bool applypost) const
{
    std::vector<double> grid = mgrid3d;
    if (applypost)
    {
        if (musecqwindow3d) applyQWindow3D(grid);

        const double rdf_scale = (mtotaloccupancy * msfaverage == 0.0) ? 0.0 :
            1.0 / (mtotaloccupancy * msfaverage * msfaverage);
        if (rdf_scale != 1.0)
            for (double& val : grid) val *= rdf_scale;

        if (mapplyrho0background3d)
        {
            const double rho0_bg = computeRho0Background();
            if (rho0_bg != 0.0)
                for (double& val : grid) val -= rho0_bg;
        }

        double qdamp = 0.0;
        try { qdamp = this->getEnvelopeByType("qresolution")->getDoubleAttr("qdamp"); }
        catch (...) { qdamp = 0.0; }
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

double PDF3DCPUCalculator::computeRho0Background() const
{
    const StructureAdapterPtr& structure = this->getStructure();
    if (!structure)  return 0.0;
    const double partialpdfscale = this->getPartialPDFScale();
    return partialpdfscale * structure->numberDensity();
}

void PDF3DCPUCalculator::applyQWindow3D(std::vector<double>& grid) const
{
    if (grid.empty())  return;

    const double qmin = this->getQmin();
    const double qmax = this->getQmax();
    if (qmin <= 0.0 && qmax <= 0.0)  return;

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

// Protected Methods ---------------------------------------------------------

void PDF3DCPUCalculator::resetValue()
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
    
    PDFCalculator::resetValue();
    if (mevaluator) mevaluator->setFlag(USEFULLSUM, true);
}

void PDF3DCPUCalculator::addPairContribution(const BaseBondGenerator& bnds, int summationscale)
{
    if (bnds.distance() == 0.0) return;

    int i0 = bnds.site0();
    int i1 = bnds.site1();
    
    if (i0 >= static_cast<int>(msfCache.size()) || i1 >= static_cast<int>(msfCache.size()))
        return;

    const double sfprod = msfCache[i0] * msfCache[i1] *
        bnds.multiplicity() * static_cast<double>(summationscale);

    const R3::Vector& rvec = bnds.r01();
    const R3::Matrix& U_i = bnds.Ucartesian0();
    const R3::Matrix& U_j = bnds.Ucartesian1();

    // Deliberately unoptimized reference path: repeat identical scalar work and
    // scale each pass so the final grid remains comparable to the optimized CPU
    // and GPU calculators while exposing a CPU baseline for comparison.
    const int repeats = std::max(1, mcpuworkfactor);
    const double scaled_sfprod = sfprod / static_cast<double>(repeats);
    for (int rep = 0; rep < repeats; ++rep)
    {
        addAnisotropicGaussianToGrid(rvec, U_i, U_j, scaled_sfprod);
    }
}

void PDF3DCPUCalculator::addAnisotropicGaussianToGrid(const R3::Vector& r_ij, const R3::Matrix& U_i, const R3::Matrix& U_j, double sfprod)
{
    // Scalar CPU implementation.
    // It avoids the optimized calculator's tiled local buffers and OpenMP
    // reduction, updating the global grid directly in one serial triple loop.
    R3::Matrix Sigma = U_i + U_j;

    R3::Vector eigenvalues;
    R3::Matrix eigenvectors;
    R3::eigen_solve_3x3(Sigma, eigenvalues, eigenvectors);

    if (eigenvalues[0] <= 1e-8) {
        return;
    }

    R3::Matrix Sigma_inv = R3::inverse(Sigma);
    const double det_Sigma = R3::determinant(Sigma);
    const double two_pi = 2.0 * M_PI;
    const double norm_factor = sfprod / (pow(two_pi, 1.5) * sqrt(det_Sigma));

    double max_sigma = 0.0;
    for (int k = 0; k < 3; ++k) {
        max_sigma = std::max(max_sigma, sqrt(eigenvalues[k]));
    }

    double cutoff_radius = 4.0 * max_sigma;
    if (cutoff_radius < mdr) cutoff_radius = mdr;
    if (cutoff_radius > 10.0) cutoff_radius = 10.0;

    const double halfspan = (mnbins / 2) * mdr;
    auto get_index_range = [&](double center_val) -> std::pair<int, int> {
        const double min_val = center_val - cutoff_radius;
        const double max_val = center_val + cutoff_radius;
        int start = static_cast<int>(ceil((min_val + halfspan) / mdr));
        int end = static_cast<int>(floor((max_val + halfspan) / mdr));
        start = std::max(0, start);
        end = std::min(mnbins - 1, end);
        return {start, end};
    };

    const std::pair<int, int> xr = get_index_range(r_ij[0]);
    const std::pair<int, int> yr = get_index_range(r_ij[1]);
    const std::pair<int, int> zr = get_index_range(r_ij[2]);
    const double cutoff_sq = 16.0;

    for (int iz = zr.first; iz <= zr.second; ++iz)
    {
        const double z_grid = iz * mdr - halfspan;
        const double dz = z_grid - r_ij[2];
        for (int iy = yr.first; iy <= yr.second; ++iy)
        {
            const double y_grid = iy * mdr - halfspan;
            const double dy = y_grid - r_ij[1];
            for (int ix = xr.first; ix <= xr.second; ++ix)
            {
                const double x_grid = ix * mdr - halfspan;
                const double dx = x_grid - r_ij[0];
                R3::Vector delta(dx, dy, dz);
                R3::Vector tmp = R3::mxvecproduct(Sigma_inv, delta);
                const double mahalanobis_sq = R3::dot(delta, tmp);
                if (mahalanobis_sq > cutoff_sq) continue;

                const double val = norm_factor * exp(-0.5 * mahalanobis_sq);
                const size_t gidx = (static_cast<size_t>(iz) * mnbins + iy) * mnbins + ix;
                mgrid3d[gidx] += val;
            }
        }
    }
}

// Private Helpers -----------------------------------------------------------

size_t PDF3DCPUCalculator::coordToIndex(double x, double y, double z) const
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

void PDF3DCPUCalculator::indexToCoord(size_t idx, double& x, double& y, double& z) const
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
DIFFPY_INSTANTIATE_SERIALIZATION(diffpy::srreal::PDF3DCPUCalculator)
