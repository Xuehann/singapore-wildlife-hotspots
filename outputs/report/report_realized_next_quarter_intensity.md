# Future Wildlife Presence Hotspots Under Realized Environment

![Future Wildlife Presence Intensity](/Users/xuehan/Downloads/Wildlife/outputs/report/figure_realized_next_quarter_intensity.png)

## Figure Caption
**Figure X. Predicted next-quarter wildlife presence hotspots under the realized-environment model for monkeys and otters in Singapore.**  
The maps show the forecasted spatial concentration of group-level wildlife presence at 500 m hex-cell resolution. Darker cells indicate higher predicted intensity, while orange outlines indicate the top predicted hotspot cells. These outputs are based on the realized-environment model, which now includes roads, waterways, parks, reserves, buildings, and subzone population as current-environment predictors. They therefore represent the main defensible forecast for near-future wildlife presence hotspots rather than a planning scenario.

## Top 10 Monkey Hotspots
These locations are translated into human-readable form using the nearest named road, nearest named waterway, nearest park or reserve, and administrative region.

| Rank | Hex ID | Predicted Intensity | Habitat | Region | Nearest Road | Dist. to Road (m) | Nearest Waterway | Dist. to Waterway (m) | Nearest Park/Reserve | Dist. to Park (m) |
|---|---|---:|---|---|---|---:|---|---:|---|---:|
| 1 | `hex_01090` | 0.506869 | urban | Central Region | Ubi Road 1 | 117 | Pelton Canal | 491 | PAYA LEBAR ROAD PARK | 572 |
| 2 | `hex_00949` | 0.437663 | urban | Central Region | Queen Street | 15 | Stamford Canal | 442 | FARQUHAR GARDEN (FCP) | 460 |
| 3 | `hex_00951` | 0.401854 | urban | Central Region | Owen Road | 12 | Kent Road drain | 232 | CITY GREEN | 404 |
| 4 | `hex_00759` | 0.388370 | urban | Central Region | Leng Kee Road | 12 | Alexandra Canal | 400 | RAIL CORRIDOR | 750 |
| 5 | `hex_01107` | 0.386657 | urban | Central Region | Ubi Avenue 1 | 85 | Geylang Canal | 918 | JLN EUNOS INTERIM PK | 670 |
| 6 | `hex_00791` | 0.379355 | urban | Central Region | Nassim Hill | 36 | Stamford Canal | 342 | SBG LC ZONE 1 (TNC) | 148 |
| 7 | `hex_01262` | 0.372596 | urban | East Region | Changi Business Park Avenue 3 | 2 | Sungei Mata Ikan | 569 | SUNBIRD CIRCLE PG | 1132 |
| 8 | `hex_00757` | 0.368085 | urban | Central Region | Telok Blangah Heights | 133 | Bukit Chermin Canal | 550 | TELOK BLANGAH HILL PARK | 155 |
| 9 | `hex_00950` | 0.354353 | urban | Central Region | Upper Dickson Road | 5 | Rochor Canal | 253 | ROWELL RD OS | 315 |
| 10 | `hex_01115` | 0.350895 | urban | North-East Region | Compassvale Walk | 43 | Sungei Punggol | 1847 | ST ANNE'S WOOD PG | 519 |

## Top 10 Otter Hotspots

| Rank | Hex ID | Predicted Intensity | Habitat | Region | Nearest Road | Dist. to Road (m) | Nearest Waterway | Dist. to Waterway (m) | Nearest Park/Reserve | Dist. to Park (m) |
|---|---|---:|---|---|---|---:|---|---:|---|---:|
| 1 | `hex_00926` | 0.197316 | urban | North-East Region | Ang Mo Kio Avenue 1 | 101 | Kallang River | 349 | BISHAN-ANG MO KIO PARK (RIVER PLAINS) | 129 |
| 2 | `hex_01072` | 0.194927 | urban | Central Region | Playfair Road | 26 | Pelton Canal | 205 | PAYA LEBAR ROAD PARK | 205 |
| 3 | `hex_00889` | 0.176961 | urban | Central Region | Thomson Road | 35 | Bukit Timah 2nd Diversion Canal | 194 | NOVENA RISE PK | 643 |
| 4 | `hex_01202` | 0.169056 | urban | East Region | East Coast Park Service Road | 5 | Bayshore Drain | 93 | EAST COAST PARK AREA F | 3 |
| 5 | `hex_01076` | 0.162581 | urban | North-East Region | Lim Ah Pin Road | 48 | Sungei Tongkang | 1923 | REALTY PARK | 495 |
| 6 | `hex_00890` | 0.161206 | urban | Central Region | Toa Payoh Rise | 25 | Bukit Timah 2nd Diversion Canal | 999 | MACRITCHIE RESERVOIR PK | 628 |
| 7 | `hex_01172` | 0.158999 | urban | East Region | Bedok North Avenue 1 | 0 | Siglap Linear Park Drain | 482 | SIGLAP LINEAR PK (JLN B'WAN-UPP EC RD) | 562 |
| 8 | `hex_01203` | 0.158725 | urban | East Region | Lucky Crescent | 9 | Bayshore Drain | 378 | SENNETT AVE OS | 118 |
| 9 | `hex_01094` | 0.155422 | urban | North-East Region | Hougang Avenue 3 | 8 | Pelton Canal | 2668 | REALTY PARK | 646 |
| 10 | `hex_01262` | 0.151986 | urban | East Region | Changi Business Park Avenue 3 | 2 | Sungei Mata Ikan | 569 | SUNBIRD CIRCLE PG | 1132 |

## Discussion
This figure demonstrates the value of moving from binary occurrence mapping to group-level intensity forecasting. Instead of asking only whether monkeys or otters may occur in a given cell, the model identifies where their presence is likely to be more concentrated in the next quarter. This makes it possible to identify future presence hotspots rather than only mapping broad potential distribution.

The realized-environment results suggest that future high-intensity activity is not limited to natural-core habitat. For both monkeys and otters, the top predicted cells are concentrated in urban or urban-edge settings, often close to named roads, canals, drains, managed park spaces, and built-up areas. Because this rerun incorporates both buildings and subzone population, the urban-context interpretation is now stronger than in the earlier proxy-only version. The pattern indicates that future wildlife presence is likely to be concentrated at the urban-natural interface rather than being restricted to a strict city-versus-reserve divide. In practical terms, the model helps identify where ecological monitoring and urban wildlife management may be most needed in the near future.

The results should still be interpreted as a prioritization tool rather than a precise movement forecast. The model does not predict individual animal trajectories, and the location descriptions above are approximate summaries based on the nearest named road, waterway, and park to each hotspot cell centroid. Nevertheless, the outputs provide a GIS-based evidence layer for identifying future concentration zones and for distinguishing whether these zones are more strongly associated with urban, natural, or mixed habitat context. Although the study does not directly model wildlife conflict, the hotspot surfaces can still provide a spatial basis for future monitoring of potential human-wildlife interaction zones if incident data become available later.

## Limitations
Several limitations should be acknowledged. First, the model predicts group-level future presence intensity rather than individual movement trajectories, so the results should be interpreted as spatial prioritization surfaces rather than precise forecasts of where specific animals will travel. Second, the input data are based on observed wildlife records, which may be affected by reporting bias, uneven observer effort, and variable detectability across space and time. Areas with more human activity may therefore have more recorded sightings even if true animal use is not proportionally higher. Third, the habitat-context classification now incorporates buildings and subzone population in addition to roads, waterways, parks, and reserves, which improves its realism, but it still should not be interpreted as an exact habitat-boundary map or a legally defined land-use classification. Fourth, `Masterplan2025` is included only as a secondary planning scenario because planned land use may not yet be fully realized on the ground. Finally, the study does not directly model wildlife conflict events because no verified conflict dataset was available, and the next-year forecasts are more uncertain than the next-quarter forecasts, since predictive error accumulates as the forecasting horizon increases.

## Notes
- Main figure file: [figure_realized_next_quarter_intensity.png](/Users/xuehan/Downloads/Wildlife/outputs/report/figure_realized_next_quarter_intensity.png)
- Source hotspot table: [top_hotspots_realized_next_quarter.csv](/Users/xuehan/Downloads/Wildlife/outputs/report/top_hotspots_realized_next_quarter.csv)
- These human-readable descriptions are based on nearest named features to hotspot-cell centroids, not exact habitat boundaries.
- The updated rerun incorporates both [Buildings.shp](/Users/xuehan/Downloads/Wildlife/Buildings/Buildings.shp) and [SubzonePopulation2019.shp](/Users/xuehan/Downloads/Wildlife/SubzonePopulation2019/SubzonePopulation2019.shp) into the realized-environment model.
