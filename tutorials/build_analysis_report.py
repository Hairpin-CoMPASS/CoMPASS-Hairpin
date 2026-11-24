"""Rebuild the figure report for the New_Reward_E3E4NLF project.

Run after the figure-producing cells in 02b and 05b have been (re)run, so
`figures/task_performance`, `figures/level_1` and the matching `results/`
stats are up to date:

    python tutorials/build_analysis_report.py
"""
from pathlib import Path

from compass_labyrinth.report import build_report

PROJECT_PATH = Path(
    r"D:\Gladstone Dropbox\Palop Lab\Nina\Experiments\For Level 1 CoMPASS States\New_Reward_E3E4NLF\compass_analysis"
)

if __name__ == "__main__":
    build_report(PROJECT_PATH)
