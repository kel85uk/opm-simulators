/*
  Copyright 2013, 2015 SINTEF ICT, Applied Mathematics.
  Copyright 2015 Andreas Lauser
  Copyright 2017 IRIS

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

#ifndef OPM_SIMULATORFULLYIMPLICITBLACKOILEBOS_HEADER_INCLUDED
#define OPM_SIMULATORFULLYIMPLICITBLACKOILEBOS_HEADER_INCLUDED

#include <opm/autodiff/SimulatorFullyImplicitBlackoilOutput.hpp>
#include <opm/autodiff/IterationReport.hpp>
#include <opm/autodiff/NonlinearSolver.hpp>
#include <opm/autodiff/BlackoilModelEbos.hpp>
#include <opm/autodiff/BlackoilModelParameters.hpp>
#include <opm/autodiff/WellStateFullyImplicitBlackoil.hpp>
#include <opm/autodiff/BlackoilWellModel.hpp>
#include <opm/autodiff/BlackoilAquiferModel.hpp>
#include <opm/autodiff/RateConverter.hpp>
#include <opm/autodiff/SimFIBODetails.hpp>
#include <opm/autodiff/moduleVersion.hpp>
#include <opm/simulators/timestepping/AdaptiveTimeStepping.hpp>
#include <opm/core/utility/initHydroCarbonState.hpp>
#include <opm/core/utility/StopWatch.hpp>

#include <opm/common/Exceptions.hpp>
#include <opm/common/ErrorMacros.hpp>

#include <dune/common/unused.hh>

namespace Opm {


/// a simulator for the blackoil model
template<class TypeTag>
class SimulatorFullyImplicitBlackoilEbos
{
public:
    typedef typename GET_PROP_TYPE(TypeTag, Simulator) Simulator;
    typedef typename GET_PROP_TYPE(TypeTag, Grid) Grid;
    typedef typename GET_PROP_TYPE(TypeTag, FluidSystem) FluidSystem;
    typedef typename GET_PROP_TYPE(TypeTag, ElementContext) ElementContext;
    typedef typename GET_PROP_TYPE(TypeTag, Indices) BlackoilIndices;
    typedef typename GET_PROP_TYPE(TypeTag, PrimaryVariables)  PrimaryVariables;
    typedef typename GET_PROP_TYPE(TypeTag, MaterialLaw) MaterialLaw;
    typedef typename GET_PROP_TYPE(TypeTag, SolutionVector)    SolutionVector ;
    typedef typename GET_PROP_TYPE(TypeTag, MaterialLawParams) MaterialLawParams;

    typedef Ewoms::BlackOilPolymerModule<TypeTag> PolymerModule;

    typedef WellStateFullyImplicitBlackoil WellState;
    typedef BlackoilState ReservoirState;
    typedef BlackoilOutputWriter OutputWriter;
    typedef BlackoilModelEbos<TypeTag> Model;
    typedef BlackoilModelParameters ModelParameters;
    typedef NonlinearSolver<Model> Solver;
    typedef BlackoilWellModel<TypeTag> WellModel;
    typedef BlackoilAquiferModel<TypeTag> AquiferModel;


    /// Initialise from parameters and objects to observe.
    /// \param[in] param       parameters, this class accepts the following:
    ///     parameter (default)            effect
    ///     -----------------------------------------------------------
    ///     output (true)                  write output to files?
    ///     output_dir ("output")          output directoty
    ///     output_interval (1)            output every nth step
    ///     nl_pressure_residual_tolerance (0.0) pressure solver residual tolerance (in Pascal)
    ///     nl_pressure_change_tolerance (1.0)   pressure solver change tolerance (in Pascal)
    ///     nl_pressure_maxiter (10)       max nonlinear iterations in pressure
    ///     nl_maxiter (30)                max nonlinear iterations in transport
    ///     nl_tolerance (1e-9)            transport solver absolute residual tolerance
    ///     num_transport_substeps (1)     number of transport steps per pressure step
    ///     use_segregation_split (false)  solve for gravity segregation (if false,
    ///                                    segregation is ignored).
    ///
    /// \param[in] props         fluid and rock properties
    /// \param[in] linsolver     linear solver
    /// \param[in] has_disgas    true for dissolved gas option
    /// \param[in] has_vapoil    true for vaporized oil option
    /// \param[in] eclipse_state the object which represents an internalized ECL deck
    /// \param[in] output_writer
    /// \param[in] threshold_pressures_by_face   if nonempty, threshold pressures that inhibit flow
    SimulatorFullyImplicitBlackoilEbos(Simulator& ebosSimulator,
                                       const ParameterGroup& param,
                                       NewtonIterationBlackoilInterface& linsolver,
                                       const bool has_disgas,
                                       const bool has_vapoil,
                                       OutputWriter& output_writer)
        : ebosSimulator_(ebosSimulator),
          param_(param),
          model_param_(param),
          solver_param_(param),
          solver_(linsolver),
          phaseUsage_(phaseUsageFromDeck(eclState())),
          has_disgas_(has_disgas),
          has_vapoil_(has_vapoil),
          terminal_output_(param.getDefault("output_terminal", true)),
          output_writer_(output_writer),
          is_parallel_run_( false )
    {
#if HAVE_MPI
        if ( solver_.parallelInformation().type() == typeid(ParallelISTLInformation) )
        {
            const ParallelISTLInformation& info =
                boost::any_cast<const ParallelISTLInformation&>(solver_.parallelInformation());
            // Only rank 0 does print to std::cout
            terminal_output_ = terminal_output_ && ( info.communicator().rank() == 0 );
            is_parallel_run_ = ( info.communicator().size() > 1 );
        }
#endif
        createLocalFipnum();
    }

    /// Run the simulation.
    /// This will run succesive timesteps until timer.done() is true. It will
    /// modify the reservoir and well states.
    /// \param[in,out] timer       governs the requested reporting timesteps
    /// \param[in,out] state       state of reservoir: pressure, fluxes
    /// \return                    simulation report, with timing data
    SimulatorReport run(SimulatorTimer& timer)
    {

        ReservoirState dummy_state(0,0,0);

        WellState prev_well_state;

        ExtraData extra;

        failureReport_ = SimulatorReport();

        if (output_writer_.isRestart()) {
            // This is a restart, populate WellState and ReservoirState state objects from restart file
            ReservoirState stateInit(Opm::UgGridHelpers::numCells(grid()),
                                     Opm::UgGridHelpers::numFaces(grid()),
                                     phaseUsage_.num_phases);
            output_writer_.initFromRestartFile(phaseUsage_, grid(), stateInit, prev_well_state, extra);
            initHydroCarbonState(stateInit, phaseUsage_, Opm::UgGridHelpers::numCells(grid()), has_disgas_, has_vapoil_);
            initHysteresisParams(stateInit);
            // communicate the restart solution to ebos
            convertInput(/*iterationIdx=*/0, stateInit, ebosSimulator_ );
            ebosSimulator_.model().invalidateIntensiveQuantitiesCache(/*timeIdx=*/0);
            // Sync the overlap region of the inital solution. It was generated
            // from the ReservoirState which has wrong values in the ghost region
            // for some models (SPE9, Norne, Model 2)
            ebosSimulator_.model().syncOverlap();
        }

        // Create timers and file for writing timing info.
        Opm::time::StopWatch solver_timer;
        Opm::time::StopWatch total_timer;
        total_timer.start();

        // adaptive time stepping
        const auto& events = schedule().getEvents();
        std::unique_ptr< AdaptiveTimeStepping > adaptiveTimeStepping;
        const bool useTUNING = param_.getDefault("use_TUNING", false);
        if( param_.getDefault("timestep.adaptive", true ) )
        {
            if (useTUNING) {
                adaptiveTimeStepping.reset( new AdaptiveTimeStepping( schedule().getTuning(), timer.currentStepNum(), param_, terminal_output_ ) );
            } else {
                adaptiveTimeStepping.reset( new AdaptiveTimeStepping( param_, terminal_output_ ) );
            }

            if (output_writer_.isRestart()) {
                if (extra.suggested_step > 0.0) {
                    adaptiveTimeStepping->setSuggestedNextStep(extra.suggested_step);
                }
            }
        }

        SimulatorReport report;
        SimulatorReport stepReport;

        WellModel well_model(ebosSimulator_, model_param_, terminal_output_);
        if (output_writer_.isRestart()) {
            well_model.setRestartWellState(prev_well_state); // Neccessary for perfect restarts
        }

        WellState wellStateDummy; //not used. Only passed to make the old interfaces happy

        AquiferModel aquifer_model(ebosSimulator_, model_param_, terminal_output_);
        // aquifer_model.hack_init(ebosSimulator_);

        // Main simulation loop.
        while (!timer.done()) {
            // Report timestep.
            if ( terminal_output_ )
            {
                std::ostringstream ss;
                timer.report(ss);
                OpmLog::debug(ss.str());
            }

            // Run a multiple steps of the solver depending on the time step control.
            solver_timer.start();

            well_model.beginReportStep(timer.currentStepNum());

            aquifer_model.beginReportStep(timer.currentStepNum());

            auto solver = createSolver(well_model, aquifer_model);

            // Compute orignal fluid in place if this has not been done yet
            if (originalFluidInPlace_.data.empty()) {
                originalFluidInPlace_ = computeFluidInPlace(*solver);
            }

            // write the inital state at the report stage
            if (timer.initialStep()) {
                Dune::Timer perfTimer;
                perfTimer.start();

                if (terminal_output_) {
                    outputFluidInPlace(timer, originalFluidInPlace_);
                }

                // No per cell data is written for initial step, but will be
                // for subsequent steps, when we have started simulating
                output_writer_.writeTimeStep( timer, dummy_state, well_model.wellState(), solver->model() );

                report.output_write_time += perfTimer.stop();
            }

            if( terminal_output_ )
            {
                std::ostringstream step_msg;
                boost::posix_time::time_facet* facet = new boost::posix_time::time_facet("%d-%b-%Y");
                step_msg.imbue(std::locale(std::locale::classic(), facet));
                step_msg << "\nReport step " << std::setw(2) <<timer.currentStepNum()
                         << "/" << timer.numSteps()
                         << " at day " << (double)unit::convert::to(timer.simulationTimeElapsed(), unit::day)
                         << "/" << (double)unit::convert::to(timer.totalTime(), unit::day)
                         << ", date = " << timer.currentDateTime();
                OpmLog::info(step_msg.str());
            }

            solver->model().beginReportStep();

            // If sub stepping is enabled allow the solver to sub cycle
            // in case the report steps are too large for the solver to converge
            //
            // \Note: The report steps are met in any case
            // \Note: The sub stepping will require a copy of the state variables
            if( adaptiveTimeStepping ) {
                if (useTUNING) {
                    if(events.hasEvent(ScheduleEvents::TUNING_CHANGE,timer.currentStepNum())) {
                        adaptiveTimeStepping->updateTUNING(schedule().getTuning(), timer.currentStepNum());
                    }
                }

                bool event = events.hasEvent(ScheduleEvents::NEW_WELL, timer.currentStepNum()) ||
                        events.hasEvent(ScheduleEvents::PRODUCTION_UPDATE, timer.currentStepNum()) ||
                        events.hasEvent(ScheduleEvents::INJECTION_UPDATE, timer.currentStepNum()) ||
                        events.hasEvent(ScheduleEvents::WELL_STATUS_CHANGE, timer.currentStepNum());
                stepReport = adaptiveTimeStepping->step( timer, *solver, dummy_state, wellStateDummy, event, output_writer_,
                                                         output_writer_.requireFIPNUM() ? &fipnum_ : nullptr );
                report += stepReport;
                failureReport_ += adaptiveTimeStepping->failureReport();
            }
            else {
                // solve for complete report step
                stepReport = solver->step(timer, dummy_state, wellStateDummy);
                report += stepReport;
                failureReport_ += solver->failureReport();

                if( terminal_output_ )
                {
                    //stepReport.briefReport();
                    std::ostringstream iter_msg;
                    iter_msg << "Stepsize " << (double)unit::convert::to(timer.currentStepLength(), unit::day);
                    if (solver->wellIterations() != 0) {
                        iter_msg << " days well iterations = " << solver->wellIterations() << ", ";
                    }
                    iter_msg << "non-linear iterations = " << solver->nonlinearIterations()
                             << ", total linear iterations = " << solver->linearIterations()
                             << "\n";
                    OpmLog::info(iter_msg.str());
                }
            }

            solver->model().endReportStep();
            aquifer_model.endReportStep();
            well_model.endReportStep();

            // take time that was used to solve system for this reportStep
            solver_timer.stop();

            // update timing.
            report.solver_time += solver_timer.secsSinceStart();

            // Increment timer, remember well state.
            ++timer;

            // Compute current fluid in place.
            const auto currentFluidInPlace = computeFluidInPlace(*solver);

            if (terminal_output_ )
            {
                outputFluidInPlace(timer, currentFluidInPlace);

                std::string msg =
                    "Time step took " + std::to_string(solver_timer.secsSinceStart()) + " seconds; "
                    "total solver time " + std::to_string(report.solver_time) + " seconds.";
                OpmLog::debug(msg);
            }

            // write simulation state at the report stage
            Dune::Timer perfTimer;
            perfTimer.start();
            const double nextstep = adaptiveTimeStepping ? adaptiveTimeStepping->suggestedNextStep() : -1.0;
            output_writer_.writeTimeStep( timer, dummy_state, well_model.wellState(), solver->model(), false, nextstep, report);
            report.output_write_time += perfTimer.stop();

        }

        // Stop timer and create timing report
        total_timer.stop();
        report.total_time = total_timer.secsSinceStart();
        report.converged = true;

        auto reportaquifer = aquifer_model.lastReport();

        return report;
    }

    /** \brief Returns the simulator report for the failed substeps of the simulation.
     */
    const SimulatorReport& failureReport() const { return failureReport_; };

    const Grid& grid() const
    { return ebosSimulator_.gridManager().grid(); }

protected:

    std::unique_ptr<Solver> createSolver(WellModel& well_model, AquiferModel& aquifer_model)
    {
        auto model = std::unique_ptr<Model>(new Model(ebosSimulator_,
                                                      model_param_,
                                                      well_model,
                                                      aquifer_model,
                                                      solver_,
                                                      terminal_output_));

        return std::unique_ptr<Solver>(new Solver(solver_param_, std::move(model)));
    }


    void createLocalFipnum()
    {
        const std::vector<int>& fipnum_global = eclState().get3DProperties().getIntGridProperty("FIPNUM").getData();
        // Get compressed cell fipnum.
        fipnum_.resize(Opm::UgGridHelpers::numCells(grid()));
        if (fipnum_global.empty()) {
            std::fill(fipnum_.begin(), fipnum_.end(), 0);
        } else {
            for (size_t c = 0; c < fipnum_.size(); ++c) {
                fipnum_[c] = fipnum_global[Opm::UgGridHelpers::globalCell(grid())[c]];
            }
        }
    }


    void FIPUnitConvert(const UnitSystem& units,
                        std::vector<std::vector<double>>& fip)
    {
        for (size_t i = 0; i < fip.size(); ++i) {
            FIPUnitConvert(units, fip[i]);
        }
    }


    void FIPUnitConvert(const UnitSystem& units,
                        std::vector<double>& fip)
    {
        if (units.getType() == UnitSystem::UnitType::UNIT_TYPE_FIELD) {
            fip[0] = unit::convert::to(fip[0], unit::stb);
            fip[1] = unit::convert::to(fip[1], unit::stb);
            fip[2] = unit::convert::to(fip[2], 1000*unit::cubic(unit::feet));
            fip[3] = unit::convert::to(fip[3], 1000*unit::cubic(unit::feet));
            fip[4] = unit::convert::to(fip[4], unit::stb);
            fip[5] = unit::convert::to(fip[5], unit::stb);
            fip[6] = unit::convert::to(fip[6], unit::psia);
        }
        else if (units.getType() == UnitSystem::UnitType::UNIT_TYPE_METRIC) {
            fip[6] = unit::convert::to(fip[6], unit::barsa);
        }
        else {
            OPM_THROW(std::runtime_error, "Unsupported unit type for fluid in place output.");
        }
    }


    std::vector<double> FIPTotals(const std::vector<std::vector<double>>& fip)
    {
        std::vector<double> totals(7,0.0);
        for (int i = 0; i < 5; ++i) {
            for (size_t reg = 0; reg < fip.size(); ++reg) {
                totals[i] += fip[reg][i];
            }
        }

        const auto& gridView = ebosSimulator_.gridManager().gridView();
        const auto& comm = gridView.comm();
        double pv_hydrocarbon_sum = 0.0;
        double p_pv_hydrocarbon_sum = 0.0;

        ElementContext elemCtx(ebosSimulator_);
        const auto& elemEndIt = gridView.template end</*codim=*/0>();
        for (auto elemIt = gridView.template begin</*codim=*/0>();
             elemIt != elemEndIt;
             ++elemIt)
        {
            const auto& elem = *elemIt;
            if (elem.partitionType() != Dune::InteriorEntity) {
                continue;
            }

            elemCtx.updatePrimaryStencil(elem);
            elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);

            const unsigned cellIdx = elemCtx.globalSpaceIndex(/*spaceIdx=*/0, /*timeIdx=*/0);
            const auto& intQuants = elemCtx.intensiveQuantities(/*spaceIdx=*/0, /*timeIdx=*/0);
            const auto& fs = intQuants.fluidState();

            const double p = fs.pressure(FluidSystem::oilPhaseIdx).value();
            const double hydrocarbon = fs.saturation(FluidSystem::oilPhaseIdx).value() + fs.saturation(FluidSystem::gasPhaseIdx).value();

            // calculate the pore volume of the current cell. Note that the
            // porosity returned by the intensive quantities is defined as the
            // ratio of pore space to total cell volume and includes all pressure
            // dependent (-> rock compressibility) and static modifiers (MULTPV,
            // MULTREGP, NTG, PORV, MINPV and friends). Also note that because of
            // this, the porosity returned by the intensive quantities can be
            // outside of the physical range [0, 1] in pathetic cases.
            const double pv =
                ebosSimulator_.model().dofTotalVolume(cellIdx)
                * intQuants.porosity().value();

            totals[5] += pv;
            pv_hydrocarbon_sum += pv*hydrocarbon;
            p_pv_hydrocarbon_sum += p*pv*hydrocarbon;
        }

        pv_hydrocarbon_sum = comm.sum(pv_hydrocarbon_sum);
        p_pv_hydrocarbon_sum = comm.sum(p_pv_hydrocarbon_sum);
        totals[5] = comm.sum(totals[5]);
        totals[6] = (p_pv_hydrocarbon_sum / pv_hydrocarbon_sum);

        return totals;
    }


    struct FluidInPlace
    {
        std::vector<std::vector<double>> data;
        std::vector<double> totals;
    };


    FluidInPlace computeFluidInPlace(const Solver& solver)
    {
        FluidInPlace fip;
        fip.data = solver.computeFluidInPlace(fipnum_);
        fip.totals = FIPTotals(fip.data);
        FIPUnitConvert(eclState().getUnits(), fip.data);
        FIPUnitConvert(eclState().getUnits(), fip.totals);
        return fip;
    }


    void outputFluidInPlace(const SimulatorTimer& timer,
                            const FluidInPlace& currentFluidInPlace)
    {
        if (!timer.initialStep()) {
            const std::string version = moduleVersionName();
            outputTimestampFIP(timer, version);
        }
        outputRegionFluidInPlace(originalFluidInPlace_.totals,
                                 currentFluidInPlace.totals,
                                 eclState().getUnits(),
                                 0);
        for (size_t reg = 0; reg < originalFluidInPlace_.data.size(); ++reg) {
            outputRegionFluidInPlace(originalFluidInPlace_.data[reg],
                                     currentFluidInPlace.data[reg],
                                     eclState().getUnits(),
                                     reg+1);
        }
    }


    void outputTimestampFIP(const SimulatorTimer& timer, const std::string version)
    {
        std::ostringstream ss;
        boost::posix_time::time_facet* facet = new boost::posix_time::time_facet("%d %b %Y");
        ss.imbue(std::locale(std::locale::classic(), facet));
        ss << "\n                              **************************************************************************\n"
        << "  Balance  at" << std::setw(10) << (double)unit::convert::to(timer.simulationTimeElapsed(), unit::day) << "  Days"
        << " *" << std::setw(30) << eclState().getTitle() << "                                          *\n"
        << "  Report " << std::setw(4) << timer.reportStepNum() << "    " << timer.currentDateTime()
        << "  *                                             Flow  version " << std::setw(11) << version << "  *\n"
        << "                              **************************************************************************\n";
        OpmLog::note(ss.str());
    }


    void outputRegionFluidInPlace(const std::vector<double>& oip, const std::vector<double>& cip, const UnitSystem& units, const int reg)
    {
        std::ostringstream ss;
        if (!reg) {
            ss << "                                                  ===================================================\n"
               << "                                                  :                   Field Totals                  :\n";
        } else {
            ss << "                                                  ===================================================\n"
               << "                                                  :        FIPNUM report region  "
               << std::setw(2) << reg << "                 :\n";
        }
        if (units.getType() == UnitSystem::UnitType::UNIT_TYPE_METRIC) {
            ss << "                                                  :      PAV  =" << std::setw(14) << cip[6] << " BARSA                 :\n"
               << std::fixed << std::setprecision(0)
               << "                                                  :      PORV =" << std::setw(14) << cip[5] << "   RM3                 :\n";
            if (!reg) {
                ss << "                                                  : Pressure is weighted by hydrocarbon pore volume :\n"
                   << "                                                  : Porv volumes are taken at reference conditions  :\n";
            }
            ss << "                         :--------------- Oil    SM3 ---------------:-- Wat    SM3 --:--------------- Gas    SM3 ---------------:\n";
        }
        if (units.getType() == UnitSystem::UnitType::UNIT_TYPE_FIELD) {
            ss << "                                                  :      PAV  =" << std::setw(14) << cip[6] << "  PSIA                 :\n"
               << std::fixed << std::setprecision(0)
               << "                                                  :      PORV =" << std::setw(14) << cip[5] << "   RB                  :\n";
            if (!reg) {
                ss << "                                                  : Pressure is weighted by hydrocarbon pore volume :\n"
                   << "                                                  : Pore volumes are taken at reference conditions  :\n";
            }
            ss << "                         :--------------- Oil    STB ---------------:-- Wat    STB --:--------------- Gas   MSCF ---------------:\n";
        }
        ss << "                         :      Liquid        Vapour        Total   :      Total     :      Free        Dissolved       Total   :" << "\n"
           << ":------------------------:------------------------------------------:----------------:------------------------------------------:" << "\n"
           << ":Currently   in place    :" << std::setw(14) << cip[1] << std::setw(14) << cip[4] << std::setw(14) << (cip[1]+cip[4]) << ":"
           << std::setw(13) << cip[0] << "   :" << std::setw(14) << (cip[2]) << std::setw(14) << cip[3] << std::setw(14) << (cip[2] + cip[3]) << ":\n"
           << ":------------------------:------------------------------------------:----------------:------------------------------------------:\n"
           << ":Originally  in place    :" << std::setw(14) << oip[1] << std::setw(14) << oip[4] << std::setw(14) << (oip[1]+oip[4]) << ":"
           << std::setw(13) << oip[0] << "   :" << std::setw(14) << oip[2] << std::setw(14) << oip[3] << std::setw(14) << (oip[2] + oip[3]) << ":\n"
           << ":========================:==========================================:================:==========================================:\n";
        OpmLog::note(ss.str());
    }


    const EclipseState& eclState() const
    { return ebosSimulator_.gridManager().eclState(); }


    const Schedule& schedule() const
    { return ebosSimulator_.gridManager().schedule(); }

    void initHysteresisParams(ReservoirState& state) {
        const int num_cells = Opm::UgGridHelpers::numCells(grid());

        typedef std::vector<double> VectorType;

        const VectorType& somax = state.getCellData( "SOMAX" );

        for (int cellIdx = 0; cellIdx < num_cells; ++cellIdx) {
            ebosSimulator_.model().setMaxOilSaturation(somax[cellIdx], cellIdx);
        }

        if (ebosSimulator_.problem().materialLawManager()->enableHysteresis()) {
            auto matLawManager = ebosSimulator_.problem().materialLawManager();

            VectorType& pcSwMdc_ow = state.getCellData( "PCSWMDC_OW" );
            VectorType& krnSwMdc_ow = state.getCellData( "KRNSWMDC_OW" );

            VectorType& pcSwMdc_go = state.getCellData( "PCSWMDC_GO" );
            VectorType& krnSwMdc_go = state.getCellData( "KRNSWMDC_GO" );

            for (int cellIdx = 0; cellIdx < num_cells; ++cellIdx) {
                matLawManager->setOilWaterHysteresisParams(
                        pcSwMdc_ow[cellIdx],
                        krnSwMdc_ow[cellIdx],
                        cellIdx);
                matLawManager->setGasOilHysteresisParams(
                        pcSwMdc_go[cellIdx],
                        krnSwMdc_go[cellIdx],
                        cellIdx);
            }
        }
    }


    // Used to convert initial Reservoirstate to primary variables in the SolutionVector
    void convertInput( const int iterationIdx,
                       const ReservoirState& reservoirState,
                       Simulator& simulator ) const
    {
        SolutionVector& solution = simulator.model().solution( 0 /* timeIdx */ );
        const Opm::PhaseUsage pu = phaseUsage_;

        const std::vector<bool> active = detail::activePhases(pu);
        bool has_solvent = GET_PROP_VALUE(TypeTag, EnableSolvent);
        bool has_polymer = GET_PROP_VALUE(TypeTag, EnablePolymer);

        const int numCells = reservoirState.numCells();
        const int numPhases = phaseUsage_.num_phases;
        const auto& oilPressure = reservoirState.pressure();
        const auto& saturations = reservoirState.saturation();
        const auto& rs          = reservoirState.gasoilratio();
        const auto& rv          = reservoirState.rv();
        for( int cellIdx = 0; cellIdx<numCells; ++cellIdx )
        {
            // set non-switching primary variables
            PrimaryVariables& cellPv = solution[ cellIdx ];
            // set water saturation
            if ( active[Water] ) {
                cellPv[BlackoilIndices::waterSaturationIdx] = saturations[cellIdx*numPhases + pu.phase_pos[Water]];
            }

            if (has_solvent) {
                cellPv[BlackoilIndices::solventSaturationIdx] = reservoirState.getCellData( reservoirState.SSOL )[cellIdx];
            }

            if (has_polymer) {
                cellPv[BlackoilIndices::polymerConcentrationIdx] = reservoirState.getCellData( reservoirState.POLYMER )[cellIdx];
            }


            // set switching variable and interpretation
            if ( active[Gas] ) {
                if( reservoirState.hydroCarbonState()[cellIdx] == HydroCarbonState::OilOnly && has_disgas_ )
                {
                    cellPv[BlackoilIndices::compositionSwitchIdx] = rs[cellIdx];
                    cellPv[BlackoilIndices::pressureSwitchIdx] = oilPressure[cellIdx];
                    cellPv.setPrimaryVarsMeaning( PrimaryVariables::Sw_po_Rs );
                }
                else if( reservoirState.hydroCarbonState()[cellIdx] == HydroCarbonState::GasOnly && has_vapoil_ )
                {
                    // this case (-> gas only with vaporized oil in the gas) is
                    // relatively expensive as it requires to compute the capillary
                    // pressure in order to get the gas phase pressure. (the reason why
                    // ebos uses the gas pressure here is that it makes the common case
                    // of the primary variable switching code fast because to determine
                    // whether the oil phase appears one needs to compute the Rv value
                    // for the saturated gas phase and if this is not available as a
                    // primary variable, it needs to be computed.) luckily for here, the
                    // gas-only case is not too common, so the performance impact of this
                    // is limited.
                    typedef Opm::SimpleModularFluidState<double,
                            /*numPhases=*/3,
                            /*numComponents=*/3,
                            FluidSystem,
                            /*storePressure=*/false,
                            /*storeTemperature=*/false,
                            /*storeComposition=*/false,
                            /*storeFugacity=*/false,
                            /*storeSaturation=*/true,
                            /*storeDensity=*/false,
                            /*storeViscosity=*/false,
                            /*storeEnthalpy=*/false> SatOnlyFluidState;
                    SatOnlyFluidState fluidState;
                    if ( active[Water] ) {
                        fluidState.setSaturation(FluidSystem::waterPhaseIdx, saturations[cellIdx*numPhases + pu.phase_pos[Water]]);
                    }
                    else {
                        fluidState.setSaturation(FluidSystem::waterPhaseIdx, 0.0);
                    }
                    fluidState.setSaturation(FluidSystem::oilPhaseIdx, saturations[cellIdx*numPhases + pu.phase_pos[Oil]]);
                    fluidState.setSaturation(FluidSystem::gasPhaseIdx, saturations[cellIdx*numPhases + pu.phase_pos[Gas]]);

                    double pC[/*numPhases=*/3] = { 0.0, 0.0, 0.0 };
                    const MaterialLawParams& matParams = simulator.problem().materialLawParams(cellIdx);
                    MaterialLaw::capillaryPressures(pC, matParams, fluidState);
                    double pg = oilPressure[cellIdx] + (pC[FluidSystem::gasPhaseIdx] - pC[FluidSystem::oilPhaseIdx]);

                    cellPv[BlackoilIndices::compositionSwitchIdx] = rv[cellIdx];
                    cellPv[BlackoilIndices::pressureSwitchIdx] = pg;
                    cellPv.setPrimaryVarsMeaning( PrimaryVariables::Sw_pg_Rv );
                }
                else
                {
                    assert( reservoirState.hydroCarbonState()[cellIdx] == HydroCarbonState::GasAndOil);
                    cellPv[BlackoilIndices::compositionSwitchIdx] = saturations[cellIdx*numPhases + pu.phase_pos[Gas]];
                    cellPv[BlackoilIndices::pressureSwitchIdx] = oilPressure[ cellIdx ];
                    cellPv.setPrimaryVarsMeaning( PrimaryVariables::Sw_po_Sg );
                }
            } else {
                // for oil-water case oil pressure should be used as primary variable
                cellPv[BlackoilIndices::pressureSwitchIdx] = oilPressure[cellIdx];
            }
        }

        // store the solution at the beginning of the time step
        if( iterationIdx == 0 )
        {
            simulator.model().solution( 1 /* timeIdx */ ) = solution;
        }
    }


    // Data.
    Simulator& ebosSimulator_;

    std::vector<int> fipnum_;
    FluidInPlace originalFluidInPlace_;

    typedef typename Solver::SolverParameters SolverParameters;

    SimulatorReport failureReport_;

    const ParameterGroup param_;
    ModelParameters model_param_;
    SolverParameters solver_param_;

    // Observed objects.
    NewtonIterationBlackoilInterface& solver_;
    PhaseUsage phaseUsage_;
    // Misc. data
    const bool has_disgas_;
    const bool has_vapoil_;
    bool       terminal_output_;
    // output_writer
    OutputWriter& output_writer_;

    // Whether this a parallel simulation or not
    bool is_parallel_run_;

};

} // namespace Opm

#endif // OPM_SIMULATORFULLYIMPLICITBLACKOIL_HEADER_INCLUDED
