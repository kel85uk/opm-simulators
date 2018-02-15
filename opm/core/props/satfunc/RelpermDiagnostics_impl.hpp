/*
  Copyright 2016 Statoil ASA.

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

#ifndef OPM_RELPERMDIAGNOSTICS_IMPL_HEADER_INCLUDED
#define OPM_RELPERMDIAGNOSTICS_IMPL_HEADER_INCLUDED

#include <vector>
#include <utility>

#include <opm/core/props/satfunc/RelpermDiagnostics.hpp>
#include <opm/grid/utility/compressedToCartesian.hpp>

namespace Opm {

    template <class GridT>
    void RelpermDiagnostics::diagnosis(const Opm::EclipseState& eclState,
                                       const Opm::Deck& deck,
                                       const GridT& grid)
    {
        OpmLog::info("\n===============Saturation Functions Diagnostics===============\n");
        phaseCheck_(eclState);
        satFamilyCheck_(eclState);
        tableCheck_(eclState);
        unscaledEndPointsCheck_(deck, eclState);
        scaledEndPointsCheck_(deck, eclState, grid);
    }

    template <class GridT>
    void RelpermDiagnostics::scaledEndPointsCheck_(const Deck& deck,
                                                   const EclipseState& eclState,
                                                   const GridT& grid)
    {
        // All end points are subject to round-off errors, checks should account for it
        const float tolerance = 1e-6;
        const int nc = Opm::UgGridHelpers::numCells(grid);
        const auto& global_cell = Opm::UgGridHelpers::globalCell(grid);
        const auto dims = Opm::UgGridHelpers::cartDims(grid);
        const auto& compressedToCartesianIdx = Opm::compressedToCartesian(nc, global_cell);
        scaledEpsInfo_.resize(nc);
        EclEpsGridProperties epsGridProperties;
        epsGridProperties.initFromDeck(deck, eclState, /*imbibition=*/false);       
        const auto& satnum = eclState.get3DProperties().getIntGridProperty("SATNUM");
        
        const std::string tag = "Scaled endpoints";
        for (int c = 0; c < nc; ++c) {
            const int cartIdx = compressedToCartesianIdx[c];
            const std::string satnumIdx = std::to_string(satnum.iget(cartIdx));
            std::array<int, 3> ijk;
            ijk[0] = cartIdx % dims[0];
            ijk[1] = (cartIdx / dims[0]) % dims[1];
            ijk[2] = cartIdx / dims[0] / dims[1];
            const std::string cellIdx = "(" + std::to_string(ijk[0]) + ", " + 
                                   std::to_string(ijk[1]) + ", " +
                                   std::to_string(ijk[2]) + ")";
            scaledEpsInfo_[c].extractScaled(eclState, epsGridProperties, cartIdx);

            // SGU <= 1.0 - SWL
            if (scaledEpsInfo_[c].Sgu > (1.0 - scaledEpsInfo_[c].Swl + tolerance)) {
                const std::string msg = "For scaled endpoints input, cell" + cellIdx + " SATNUM = " + satnumIdx + ", SGU exceed 1.0 - SWL";
                OpmLog::warning(tag, msg);
            }
            
            // SGL <= 1.0 - SWU
            if (scaledEpsInfo_[c].Sgl > (1.0 - scaledEpsInfo_[c].Swu + tolerance)) {
                const std::string msg = "For scaled endpoints input, cell" + cellIdx + " SATNUM = " + satnumIdx + ", SGL exceed 1.0 - SWU";
                OpmLog::warning(tag, msg);
            }

            if (deck.hasKeyword("SCALECRS") && fluidSystem_ == FluidSystem::BlackOil) {
                // Mobilility check.
		    if ((scaledEpsInfo_[c].Sowcr + scaledEpsInfo_[c].Swcr) >= (1.0 + tolerance)) {
                    const std::string msg = "For scaled endpoints input, cell" + cellIdx + " SATNUM = " + satnumIdx + ", SOWCR + SWCR exceed 1.0";
                    OpmLog::warning(tag, msg);
                }

            if ((scaledEpsInfo_[c].Sogcr + scaledEpsInfo_[c].Sgcr + scaledEpsInfo_[c].Swl) >= (1.0 + tolerance)) {
                    const std::string msg = "For scaled endpoints input, cell" + cellIdx + " SATNUM = " + satnumIdx + ", SOGCR + SGCR + SWL exceed 1.0";
                    OpmLog::warning(tag, msg);
                }
            }
        } 
    }

} //namespace Opm

#endif // OPM_RELPERMDIAGNOSTICS_IMPL_HEADER_INCLUDED
