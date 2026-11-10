import csv
import tempfile
import unittest

from elevator_sim import SimulationConfig
from sensitivity_analysis import (
    build_sensitivity_cases,
    run_sensitivity_analysis,
    summarize_sensitivity,
    write_sensitivity_csv,
)


class SensitivityAnalysisTests(unittest.TestCase):
    def test_case_design_covers_demand_and_physical_assumptions(self) -> None:
        cases = build_sensitivity_cases(SimulationConfig(duration_seconds=60))
        self.assertEqual(len(cases), 19)
        self.assertEqual(
            sum(case.analysis_dimension == "demand_rate" for case in cases),
            5,
        )
        self.assertEqual(
            sum(
                case.analysis_dimension == "physical_assumption"
                for case in cases
            ),
            14,
        )
        self.assertTrue(
            all(
                len(case.strategies) == 3
                for case in cases
                if case.analysis_dimension == "demand_rate"
            )
        )

    def test_small_analysis_has_ranks_references_and_complete_csv(self) -> None:
        observations = run_sensitivity_analysis(
            base_config=SimulationConfig(duration_seconds=30),
            runs=1,
            seed=101,
            traffic_patterns=("up_peak",),
        )
        self.assertEqual(len(observations), 29)
        summary = summarize_sensitivity(observations)
        self.assertEqual(len(summary), 29)

        demand_groups: dict[str, list[int]] = {}
        for record in summary:
            if record["analysis_dimension"] == "demand_rate":
                demand_groups.setdefault(str(record["scenario"]), []).append(
                    int(record["rank_by_mean_p95"])
                )
            if record["scenario"] in {"demand_0_12", "baseline"}:
                self.assertAlmostEqual(
                    float(record["p95_change_vs_reference_pct"]), 0.0
                )
        self.assertTrue(
            all(sorted(ranks) == [1, 2, 3] for ranks in demand_groups.values())
        )

        with tempfile.TemporaryDirectory() as directory:
            raw_path, summary_path = write_sensitivity_csv(
                observations, directory
            )
            with raw_path.open(encoding="utf-8", newline="") as stream:
                raw_records = list(csv.DictReader(stream))
            with summary_path.open(encoding="utf-8", newline="") as stream:
                summary_records = list(csv.DictReader(stream))
            self.assertEqual(len(raw_records), 29)
            self.assertEqual(len(summary_records), 29)
            self.assertEqual(raw_records[0]["floor_height_m"], "3.5")
            self.assertTrue(
                all(record["reference_scenario"] for record in summary_records)
            )


if __name__ == "__main__":
    unittest.main()
