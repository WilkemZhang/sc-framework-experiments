# SC Framework numerical experiments

Experiments for *Linear Convergence of Adaptive Accelerated Gradient Methods: A Unified Framework*. The five Python files reproduce the calculations and figures from the authors' Colab notebook.

## Figures from the full run

- [Convergence curves](figures/convergence/all_families_grid.png)
- Median iterations by condition ratio at [1e-8](figures/iterations_by_kappa/median_iterations_1e-8.png), [1e-12](figures/iterations_by_kappa/median_iterations_1e-12.png), and [1e-15](figures/iterations_by_kappa/median_iterations_1e-15.png).

The PDF versions are in the same folders. The [results](results/) folder contains the run records, instance parameters, reference points, and convergence trajectories from this run.

## Project files

| File | Purpose |
| --- | --- |
| `problems.py` | Generates the four problem families and defines their gradients, Bregman divergences, and reference solutions. The experiment uses five draws, seven condition ratios, and three error targets. |
| `solvers.py` | Implements the initialization, stepsizes, and updates for the seven gradient methods. |
| `plot_convergence.py` | Plots the relative objective error and relative squared distance on the first draw of each family at a condition ratio of one million. |
| `plot_iterations_by_kappa.py` | Plots median stopping iterations against the condition ratio for each error target. The shaded bands give the first and third quartiles over the five draws. |
| `main.py` | Runs the experiments or reads saved data, generates the figures, and packages the outputs. |

## Installation and use

Use Python 3.11. Open the folder containing `main.py` in PyCharm, or use that folder as your terminal's working directory. Install the dependencies with

```bash
python -m pip install -r requirements.txt
```

Choose a mode near the top of `main.py` before running it.

- **Redraw from saved data.** Set `MODE = "reference"`. The included `reference_runs.csv` contains 980 run records and `curves.npz` contains 28 trajectories. Together they reproduce all four figures. The CSV alone is sufficient for the three iteration-count figures.
- **Run the full experiment.** The default `MODE = "full"` computes all 980 runs and saves the raw results and figures.

Run `main.py` in PyCharm or enter

```bash
python main.py
```

## Outputs

The program saves each figure as PDF and PNG under `output/<mode>/figures/`. The convergence figure is named `all_families_grid`. The three iteration-count figures are named `median_iterations_1e-8`, `median_iterations_1e-12`, and `median_iterations_1e-15`. Each reports the first iteration at which both relative errors meet the indicated target. The program also writes `sc_framework_outputs.zip` in the project root.

## License

Copyright (c) 2026 Ming Yan and Weihe Zhang. Released under the [MIT License](LICENSE).
