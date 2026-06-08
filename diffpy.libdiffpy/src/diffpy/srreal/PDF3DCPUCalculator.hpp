/*************************************************************************************
*
* class PDF3DCPUCalculator -- CPU 3D real-space PDF calculator
*
*************************************************************************************/

#ifndef PDF3DCPUCalculator_HPP_INCLUDED
#define PDF3DCPUCalculator_HPP_INCLUDED

#include <diffpy/srreal/PDFCalculator.hpp>
#include <vector>
#include <string>
#include <diffpy/srreal/R3linalg.hpp>      // For R3::Matrix and R3::Vector

namespace diffpy {
namespace srreal {

class PDF3DCPUCalculator : public PDFCalculator
{
public:
    // constructor
    PDF3DCPUCalculator();

    // Public interface to retrieve 3D PDF data
    // Returns a flat vector of [x0, y0, z0, G0, x1, y1, z1, G1, ...]
    QuantityType get3DPDF() const;

    // Export dense 3D grid (nz, ny, nx) directly to binary file.
    // usefloat32=true writes float32, otherwise float64.
    // applypost=true applies q-window/rdf_scale/rho0/qdamp before export.
    void exportGrid3DBinary(const std::string& path, bool usefloat32 = true, bool applypost = true) const;

    // Grid configuration
    void setGridStep(double dr);
    double getGridStep() const;

    // Blocked accumulation configuration
    void setAccumBlockSize(int bs);
    int getAccumBlockSize() const;

    void setCPUWorkFactor(int factor);
    int getCPUWorkFactor() const;

    void setApplyRho0Background3D(bool);
    bool getApplyRho0Background3D() const;

    void setUseCQWindow3D(bool v);
    bool getUseCQWindow3D() const;

protected:
    // Override PairQuantity virtual methods
    virtual void resetValue() override;
    virtual void addPairContribution(const BaseBondGenerator& bnds, int) override;

private:
    // Helper to add anisotropic Gaussian to the grid
    void addAnisotropicGaussianToGrid(const R3::Vector& r_ij, const R3::Matrix& U_i, const R3::Matrix& U_j, double sfprod);

    // Helper: map 3D position to linear index
    size_t coordToIndex(double x, double y, double z) const;
    void indexToCoord(size_t idx, double& x, double& y, double& z) const;

    void applyQWindow3D(std::vector<double>& grid) const;
    double computeRho0Background() const;

    // Data members
    double mdr;                      // grid spacing (assumes cubic grid centered at origin)
    int mnbins;                      // number of bins per dimension (odd number, center at 0)
    std::vector<double> mgrid3d;     // flattened 3D histogram (size = mnbins^3)
    int maccumblocksize;              // retained for API compatibility
    int mcpuworkfactor;               // repeat count for CPU reference work

    bool mapplyrho0background3d;
    bool musecqwindow3d;

    std::vector<double> msfCache;
    double mtotaloccupancy;
    double msfaverage;

    // Internal constants
    static constexpr double DEFAULT_RMAX_3D = 10.0;  // 脜
    static constexpr double DEFAULT_GRID_STEP = 0.1; // 脜
};

}   // namespace srreal
}   // namespace diffpy

#endif  // PDF3DCPUCalculator_HPP_INCLUDED
