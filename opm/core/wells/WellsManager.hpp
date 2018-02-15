/*
  Copyright 2012 SINTEF ICT, Applied Mathematics.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_WELLSMANAGER_HEADER_INCLUDED
#define OPM_WELLSMANAGER_HEADER_INCLUDED

#include <unordered_set>

#include <opm/parser/eclipse/EclipseState/EclipseState.hpp>

#include <opm/core/wells/WellCollection.hpp>
#include <opm/core/wells/WellsGroup.hpp>
#include <opm/core/wells/DynamicListEconLimited.hpp>
#include <opm/parser/eclipse/EclipseState/Schedule/GroupTree.hpp>

#include <opm/grid/utility/CompressedPropertyAccess.hpp>

struct Wells;
struct UnstructuredGrid;


namespace Opm
{

    class Schedule;

    struct WellData
    {
        WellType type;
        bool allowCrossFlow;
        // WellControlType control;
        // double target;
        double reference_bhp_depth;
        // Opm::InjectionSpecification::InjectorType injected_phase;
        int welspecsline;
    };


    struct PerfData
    {
        int cell;
        double well_index;
        int satnumid;
    };
    /// This class manages a Wells struct in the sense that it
    /// encapsulates creation and destruction of the wells
    /// data structure.
    /// The resulting Wells is available through the c_wells() method.
    class WellsManager
    {
    public:
        /// Default constructor -- no wells.
        WellsManager();

        /// Construct from existing wells object.
        /// WellsManager is not properly initialised in the sense that the logic to
        /// manage control switching does not exist.
        ///
        /// @param[in] W Existing wells object.
        explicit WellsManager(struct Wells* W);

        /// Construct from input deck and grid.
        /// The permeability argument may be zero if the input contain
        /// well productivity indices, otherwise it must be given in
        /// order to approximate these by the Peaceman formula.
        ///
        /// \param deactivated_wells A set of wells that should be treated
        ///        like shut wells. E.g. in a a parallel run these would be
        ///        the wells handeled by another process. Defaults to empty set.
        template<class F2C, class FC>
        WellsManager(const Opm::EclipseState& eclipseState,
                     const Opm::Schedule& schedule,
                     const size_t timeStep,
                     int num_cells,
                     const int* global_cell,
                     const int* cart_dims,
                     int dimensions,
                     const F2C& f2c,
                     FC begin_face_centroids,
                     const DynamicListEconLimited& list_econ_limited,
                     bool is_parallel_run=false,
                     const std::unordered_set<std::string>& deactivated_wells = std::unordered_set<std::string> ());

        WellsManager(const Opm::EclipseState& eclipseState,
                     const Opm::Schedule& schedule,
                     const size_t timeStep,
                     const UnstructuredGrid& grid);
        /// Destructor.
        ~WellsManager();

        /// Does the "deck" define any wells?
        bool empty() const;

        /// Access the managed Wells.
        /// The method is named similarly to c_str() in std::string,
        /// to make it clear that we are returning a C-compatible struct.
        const Wells* c_wells() const;

        /// Access the well group hierarchy.
        const WellCollection& wellCollection() const;
        WellCollection& wellCollection();

        /// Checks if each condition is met, applies well controls where needed
        /// (that is, it either changes the active control of violating wells, or shuts
        /// down wells). Only one change is applied per invocation. Typical use will be
        /// \code
        /// solve_pressure();
        /// while(!wells.conditionsMet(...)) {
        ///     solve_pressure();
        /// }
        /// \endcode
        /// \param[in]    well_bhp  A vector containing the bhp for each well. Is assumed
        ///                         to be ordered the same way as the related Wells-struct.
        /// \param[in]    well_reservoirrates_phase
        ///                         A vector containing reservoir rates by phase for each well.
        ///                         Is assumed to be ordered the same way as the related Wells-struct,
        ///                         with all phase rates of a single well adjacent in the array.
        /// \param[in]    well_surfacerates_phase
        ///                         A vector containing surface rates by phase for each well.
        ///                         Is assumed to be ordered the same way as the related Wells-struct,
        ///                         with all phase rates of a single well adjacent in the array.
        /// \return true if no violations were found, false otherwise (false also implies a change).
        bool conditionsMet(const std::vector<double>& well_bhp,
                           const std::vector<double>& well_reservoirrates_phase,
                           const std::vector<double>& well_surfacerates_phase);

        /// Applies explicit reinjection controls. This must be called at each timestep to be correct.
        /// \param[in]    well_reservoirrates_phase
        ///                         A vector containing reservoir rates by phase for each well.
        ///                         Is assumed to be ordered the same way as the related Wells-struct,
        ///                         with all phase rates of a single well adjacent in the array.
        /// \param[in]    well_surfacerates_phase
        ///                         A vector containing surface rates by phase for each well.
        ///                         Is assumed to be ordered the same way as the related Wells-struct,
        ///                         with all phase rates of a single well adjacent in the array.
        void applyExplicitReinjectionControls(const std::vector<double>& well_reservoirrates_phase,
                                              const std::vector<double>& well_surfacerates_phase);


    private:
        template<class C2F, class FC>
        void init(const Opm::EclipseState& eclipseState,
                  const Opm::Schedule& schedule,
                  const size_t timeStep,
                  int num_cells,
                  const int* global_cell,
                  const int* cart_dims,
                  int dimensions,
                  const C2F& cell_to_faces,
                  FC begin_face_centroids,
                  const DynamicListEconLimited& list_econ_limited,
                  const std::unordered_set<std::string>& deactivated_wells);
        // Disable copying and assignment.
        WellsManager(const WellsManager& other);
        WellsManager& operator=(const WellsManager& other);
        static void setupCompressedToCartesian(const int* global_cell, int number_of_cells, std::map<int,int>& cartesian_to_compressed );
        void setupWellControls(std::vector<const Well*>& wells, size_t timeStep,
                               std::vector<std::string>& well_names, const PhaseUsage& phaseUsage,
                               const std::vector<int>& wells_on_proc,
                               const DynamicListEconLimited& list_econ_limited);

        template<class C2F, class FC, class NTG>
        void createWellsFromSpecs( std::vector<const Well*>& wells, size_t timeStep,
                                   const C2F& cell_to_faces,
                                   const int* cart_dims,
                                   FC begin_face_centroids,
                                   int dimensions,
                                   std::vector<double>& dz,
                                   std::vector<std::string>& well_names,
                                   std::vector<WellData>& well_data,
                                   std::map<std::string, int> & well_names_to_index,
                                   const PhaseUsage& phaseUsage,
                                   const std::map<int,int>& cartesian_to_compressed,
                                   const double* permeability,
                                   const NTG& ntg,
                                   std::vector<int>& wells_on_proc,
                                   const std::unordered_set<std::string>& deactivated_wells,
                                   const DynamicListEconLimited& list_econ_limited);

        void setupGuideRates(std::vector<const Well*>& wells, const size_t timeStep, std::vector<WellData>& well_data, std::map<std::string, int>& well_names_to_index);

        // Data
        Wells* w_;
        WellCollection well_collection_;
        // Whether this is a parallel simulation
        bool is_parallel_run_;
    };

} // namespace Opm

#include "WellsManager_impl.hpp"
#endif // OPM_WELLSMANAGER_HEADER_INCLUDED
