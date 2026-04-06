# Singapore Wildlife Presence Hotspots

This repository contains a GIS-based workflow for forecasting future presence hotspots of monkeys and otters in Singapore.

## Scope

The project predicts group-level wildlife presence intensity rather than individual animal trajectories. The main output is a realized-environment next-quarter hotspot forecast, with planned-land-use runs included as secondary scenario comparisons.

## Included Files

- `wildlife_movement_model.py`: end-to-end modeling workflow
- `requirements.txt`: Python dependencies
- `outputs/run_summary.json`: compact run summary
- `outputs/monkeys/monkeys_metrics.json`: monkey model metrics
- `outputs/otters/otters_metrics.json`: otter model metrics
- `outputs/report/figure_realized_next_quarter_intensity.png`: report-ready figure
- `outputs/report/top_hotspots_realized_next_quarter.csv`: human-readable Top 10 hotspot table
- `outputs/report/report_realized_next_quarter_intensity.md`: report-ready figure caption, discussion, and limitations

## Data Availability

Raw GIS datasets are not included in this public repository. They were sourced separately and excluded from version control because of size and sharing constraints.

## Main Result

The main defensible result is the realized-environment next-quarter forecast: monkeys and otters are predicted to occur more frequently in urban and urban-edge parts of Singapore, especially where roads, waterways, parks, and built-up environments overlap.
