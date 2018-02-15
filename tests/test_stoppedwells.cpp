/*
  Copyright 2014 IRIS
  
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

#include <config.h>

#define BOOST_TEST_MODULE StoppedWellsTests
#include <boost/test/unit_test.hpp>

#include <opm/parser/eclipse/Parser/Parser.hpp>
#include <opm/parser/eclipse/Parser/ParseContext.hpp>
#include <opm/parser/eclipse/EclipseState/Schedule/Schedule.hpp>

#include <opm/core/wells/WellsManager.hpp>
#include <opm/core/wells.h>
#include <opm/core/well_controls.h>
#include <opm/core/simulator/BlackoilState.hpp>
#include <opm/core/simulator/WellState.hpp>
#include <opm/grid/GridManager.hpp>


using namespace Opm;

BOOST_AUTO_TEST_CASE(TestStoppedWells)
{
    const std::string filename = "wells_stopped.data";
    Opm::ParseContext parseContext;
    Opm::Parser parser;
    Opm::Deck deck(parser.parseFile(filename , parseContext));
    Opm::EclipseState eclipseState(deck , parseContext);
    Opm::GridManager vanguard(eclipseState.getInputGrid());
    const auto& grid = eclipseState.getInputGrid();
    const TableManager table ( deck );
    const Eclipse3DProperties eclipseProperties ( deck , table, grid);
    const Schedule sched(deck, grid, eclipseProperties, Phases(true, true, true), parseContext );


    double target_surfacerate_inj;
    double target_surfacerate_prod;

    const std::vector<double> pressure = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    BlackoilState state( pressure.size() , 0 , 3);
    state.pressure() = pressure;

    // Both wells are open in the first schedule step
    {
    Opm::WellsManager wellsManager(eclipseState , sched, 0, *vanguard.c_grid());
    const Wells* wells = wellsManager.c_wells();
    const struct WellControls* ctrls0 = wells->ctrls[0];
    const struct WellControls* ctrls1 = wells->ctrls[1];
    BOOST_CHECK(well_controls_well_is_open(ctrls0));
    BOOST_CHECK(well_controls_well_is_open(ctrls1));

    target_surfacerate_inj = well_controls_iget_target(ctrls0 , 0);
    target_surfacerate_prod = well_controls_iget_target(ctrls1 , 0);

    WellState wellstate;
    wellstate.init(wells, state);
    const std::vector<double> wellrates = wellstate.wellRates();
    BOOST_CHECK_EQUAL (target_surfacerate_inj, wellrates[2]); // Gas injector
    BOOST_CHECK_EQUAL (target_surfacerate_prod, wellrates[4]); // Oil target rate
    }


    // The injector is stopped
    {
    Opm::WellsManager wellsManager(eclipseState, sched, 1 , *vanguard.c_grid());
    const Wells* wells = wellsManager.c_wells();
    const struct WellControls* ctrls0 = wells->ctrls[0];
    const struct WellControls* ctrls1 = wells->ctrls[1];
    BOOST_CHECK(well_controls_well_is_stopped(ctrls0)); // injector is stopped
    BOOST_CHECK(well_controls_well_is_open(ctrls1));

    WellState wellstate;
    wellstate.init(wells, state);

    const std::vector<double> wellrates = wellstate.wellRates();
    BOOST_CHECK_EQUAL (0, wellrates[2]); // Gas injector
    BOOST_CHECK_EQUAL (target_surfacerate_prod, wellrates[4]); // Oil target rate
    }

}
