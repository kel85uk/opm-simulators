/*
  Copyright (c) 2017 IRIS AS

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
#ifndef OPM_BLACKOILOUTPUTEBOS_HEADER_INCLUDED
#define OPM_BLACKOILOUTPUTEBOS_HEADER_INCLUDED


#include <ebos/eclproblem.hh>
#include <ewoms/common/start.hh>

#include <opm/grid/UnstructuredGrid.h>
#include <opm/simulators/timestepping/SimulatorTimerInterface.hpp>
#include <opm/core/utility/DataMap.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/core/utility/miscUtilities.hpp>
#include <opm/common/utility/parameters/ParameterGroup.hpp>
#include <opm/core/wells/DynamicListEconLimited.hpp>
#include <opm/core/simulator/SimulatorReport.hpp>

#include <opm/output/data/Cells.hpp>
#include <opm/output/data/Solution.hpp>

#include <opm/autodiff/GridHelpers.hpp>
#include <opm/autodiff/ParallelDebugOutput.hpp>
#include <opm/autodiff/Compat.hpp>

#include <opm/autodiff/WellStateFullyImplicitBlackoil.hpp>
#include <opm/autodiff/ThreadHandle.hpp>

#include <opm/parser/eclipse/EclipseState/EclipseState.hpp>
#include <opm/parser/eclipse/EclipseState/SummaryConfig/SummaryConfig.hpp>
#include <opm/parser/eclipse/EclipseState/InitConfig/InitConfig.hpp>

#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <map>

#include <boost/filesystem.hpp>

#ifdef HAVE_OPM_GRID
#include <opm/grid/CpGrid.hpp>
#endif
namespace Opm
{


    /// Extra data to read/write for OPM restarting
    struct ExtraData
    {
        double suggested_step = -1.0;
    };


    /** \brief Wrapper ECL output. */
    template<class TypeTag>
    class BlackoilOutputEbos
    {
    public:

        typedef typename GET_PROP_TYPE(TypeTag, Simulator) Simulator;
        typedef typename GET_PROP_TYPE(TypeTag, Grid) Grid;
        typedef typename GET_PROP_TYPE(TypeTag, FluidSystem) FluidSystem;
        typedef typename GET_PROP_TYPE(TypeTag, ElementContext) ElementContext;
        // constructor creating different sub writers
        BlackoilOutputEbos(Simulator& ebosSimulator,
                           const ParameterGroup& param)
            : output_( [ &param ] () -> bool {
                    // If output parameter is true or all, then we do output
                    const std::string outputString = param.getDefault("output", std::string("all"));
                    return ( outputString == "all" ||  outputString == "true" );
                }()
                ),
            ebosSimulator_(ebosSimulator),
            phaseUsage_(phaseUsageFromDeck(eclState())),
            parallelOutput_( output_ ? new ParallelDebugOutput< Grid >( grid(), eclState(), schedule(), phaseUsage_.num_phases, phaseUsage_ ) : 0 ),
            restart_double_si_( output_ ? param.getDefault("restart_double_si", false) : false ),
            asyncOutput_()
        {
            // For output.
            if ( output_ )
            {

                // create output thread if enabled and rank is I/O rank
                // async output is enabled by default if pthread are enabled
    #if HAVE_PTHREAD
                const bool asyncOutputDefault = true;
    #else
                const bool asyncOutputDefault = false;
    #endif
                if( param.getDefault("async_output", asyncOutputDefault ) )
                {
                    const bool isIORank = parallelOutput_ ? parallelOutput_->isIORank() : true;
    #if HAVE_PTHREAD
                    asyncOutput_.reset( new ThreadHandle( isIORank ) );
    #else
                    OPM_THROW(std::runtime_error,"Pthreads were not found, cannot enable async_output");
    #endif
                }
            }
        }




        /*!
         * \brief Write a blackoil reservoir state to disk for later inspection with
         *        visualization tools like ResInsight. This function will extract the
         *        requested output cell properties specified by the RPTRST keyword
         *        and write these to file.
         */
        template<class SimulationDataContainer, class Model>
        void writeTimeStep(const SimulatorTimerInterface& timer,
                           const SimulationDataContainer& reservoirStateDummy,
                           const Opm::WellStateFullyImplicitBlackoil& /*wellStateDummy*/,
                           const Model& physicalModel,
                           const bool substep = false,
                           const double nextstep = -1.0,
                           const SimulatorReport& simulatorReport = SimulatorReport())
        {
            if( output_ )
            {
                // Add TCPU if simulatorReport is not defaulted.
                const double totalSolverTime = simulatorReport.solver_time;

                const Opm::WellStateFullyImplicitBlackoil& localWellState = physicalModel.wellModel().wellState();

                if( parallelOutput_ && parallelOutput_->isParallel() )
                {
                    // If this is not the initial write and no substep, then the well
                    // state used in the computation is actually the one of the last
                    // step. We need that well state for the gathering. Otherwise
                    // It an exception with a message like "global state does not
                    // contain well ..." might be thrown.
                    // The distribution of data::solution is not done here
                    data::Solution localCellDataDummy{};
                    int wellStateStepNumber = ( ! substep && timer.reportStepNum() > 0) ?
                                (timer.reportStepNum() - 1) : timer.reportStepNum();
                    // collect all solutions to I/O rank
                    parallelOutput_->collectToIORank( reservoirStateDummy, localWellState,
                                                      localCellDataDummy,
                                                      wellStateStepNumber );
                    // Note that at this point the extraData are assumed to be global, i.e. identical across all processes.
                }

                const WellStateFullyImplicitBlackoil& wellState  = (parallelOutput_ && parallelOutput_->isParallel() ) ? parallelOutput_->globalWellState() : localWellState;

                // The writeOutput expects a local data::solution vector and a global data::well vector.
            ebosSimulator_.problem().writeOutput( wellState.report(phaseUsage_), timer.simulationTimeElapsed(), substep, totalSolverTime, nextstep);
            }
        }

        template <class SimulationDataContainer, class WellState>
        void initFromRestartFile(const PhaseUsage& /*phaseUsage*/,
                                 const Grid& /*grid */,
                                 SimulationDataContainer& simulatorstate,
                                 WellState& wellstate,
                                 ExtraData& extra)   {

            std::map<std::string, bool> extra_keys {
                {"OPMEXTRA" , false}
            };

            // gives a dummy dynamic_list_econ_limited
            DynamicListEconLimited dummy_list_econ_limited;
            const auto& defunct_well_names = ebosSimulator_.vanguard().defunctWellNames();
            WellsManager wellsmanager(eclState(),
                                      schedule(),
                                      eclState().getInitConfig().getRestartStep(),
                                      Opm::UgGridHelpers::numCells(grid()),
                                      Opm::UgGridHelpers::globalCell(grid()),
                                      Opm::UgGridHelpers::cartDims(grid()),
                                      Opm::UgGridHelpers::dimensions(grid()),
                                      Opm::UgGridHelpers::cell2Faces(grid()),
                                      Opm::UgGridHelpers::beginFaceCentroids(grid()),
                                      dummy_list_econ_limited,
                                      grid().comm().size() > 1,
                                      defunct_well_names);

            const Wells* wells = wellsmanager.c_wells();

            std::map<std::string, RestartKey> solution_keys {};
            auto restart_values = ebosSimulator_.problem().eclIO().loadRestart(solution_keys, extra_keys);

            const int nw = wells->number_of_wells;
            if (nw > 0) {
                wellstate.resize(wells, simulatorstate, phaseUsage_ ); //Resize for restart step
                wellsToState( restart_values.wells, phaseUsage_, wellstate );
            }

            const auto opmextra_iter = restart_values.extra.find("OPMEXTRA");
            if (opmextra_iter != restart_values.extra.end()) {
                std::vector<double> opmextra = opmextra_iter->second;
                assert(opmextra.size() == 1);
                extra.suggested_step = opmextra[0];
            } else {
                OpmLog::warning("Restart data is missing OPMEXTRA field, restart run may deviate from original run.");
                extra.suggested_step = -1.0;
            }
        }

        const Grid& grid()
        { return ebosSimulator_.vanguard().grid(); }

        const Schedule& schedule() const
        { return ebosSimulator_.vanguard().schedule(); }

        const EclipseState& eclState() const
        { return ebosSimulator_.vanguard().eclState(); }

        bool isRestart() const {
            const auto& initconfig = eclState().getInitConfig();
            return initconfig.restartRequested();
        }   

    protected:
        const bool output_;
        Simulator& ebosSimulator_;
        Opm::PhaseUsage phaseUsage_;
        std::unique_ptr< ParallelDebugOutputInterface > parallelOutput_;
        const bool restart_double_si_;
        std::unique_ptr< ThreadHandle > asyncOutput_;
    };





}
#endif
