# Data

`workload/hourly_aggregated_utilization.csv` is the 744-hour (31-day) processed
compute-utilization trace used by every multiday experiment. The study scripts
scale its MW column by 6.1 to form the fixed-load input for the 100 MW data
center cases. SHA-256:

`c6ed7d959d5031036a1bb510f1bb137bfb4957182d1381bedca6177c0ab54935`

The workload profile is derived from the trace described by Sakalkar et al.,
“Data Center Power Oversubscription with a Medium Voltage Power Plane and
Priority-Aware Capping,” ASPLOS 2020.

The CAISO flexible-ramping and PJM reserve observations are distributed in the
benchmark as the empirical summary statistics and hourly profile parameters in
`scripts/real_market_data_case_vcc.jl`. This keeps the benchmark input small
while exactly preserving the calibrated inputs used in the reported runs. The
source databases are [CAISO OASIS](https://oasis.caiso.com/mrioasis/logon.do)
and [PJM Data Miner 2](https://dataminer2.pjm.com/list), accessed in 2025.

The processed inputs in this directory are provided for research
reproducibility. Users remain responsible for complying with the terms of the
original data sources.
