"""Generates the 5 deliverable notebooks from the src modules, then they are
executed via nbconvert. Thin wrappers: narrative + calls into src + display."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PATH_SETUP = "import sys, os\nsys.path.insert(0, os.path.abspath('..'))\n"

NB = {
    "01_preprocess.ipynb": [
        ("md", "# 01 · Preprocessing & Unified Base\n\n"
               "Both datasets are stacked into **one** hourly long-format dataframe "
               "(`outputs/unified_panel.csv`), tagged by `source` and sliced downstream. "
               "ACN times are parsed GMT→Los Angeles; UrbanEV wide matrices are reshaped "
               "long and aggregated 5-min→hourly."),
        ("code", PATH_SETUP + "from src.data import build_unified, missing_value_log\n"
                 "panel, sessions = build_unified()\n"
                 "print('unified panel:', panel.shape)\n"
                 "panel.groupby('source').size()"),
        ("md", "### Missing-value & cleaning decisions (documented)"),
        ("code", "missing_value_log"),
        ("code", "panel.groupby('source')[['utilization','price_per_kwh','energy_kwh']].describe().T.round(3)"),
    ],
    "02_eda.ipynb": [
        ("md", "# 02 · Exploratory Data Analysis\n\n"
               "Every figure is tied to a pricing implication. Key finding: in Shenzhen, "
               "charging **occupancy peaks overnight**, not at commute hours — so peak "
               "windows must come from charging data, not traffic intuition. The Tariff "
               "Agent therefore prices on *realized* utilization, not the clock."),
        ("code", PATH_SETUP + "from src.eda import run_eda\nsummary = run_eda()\nsummary"),
        ("md", "### Figures\nSaved to `outputs/figures/`."),
        ("code", "from IPython.display import Image, display\nimport glob\n"
                 "for f in sorted(glob.glob('../outputs/figures/*.png')):\n    display(Image(f))"),
    ],
    "03_demand_model.ipynb": [
        ("md", "# 03 · Demand Prediction Agent\n\n"
               "Per-source gradient boosting forecasts utilization with strict temporal "
               "splits (no leakage), scored against a seasonal-naive baseline. "
               "Outputs: predicted utilization, expected load, congestion probability."),
        ("code", PATH_SETUP + "from src.data import load_unified\nfrom src.agents.demand import DemandAgent\n"
                 "panel = load_unified()\nagent = DemandAgent()\nmetrics = agent.fit_eval(panel)\nmetrics.round(4)"),
        ("md", "Model beats the seasonal-naive baseline on both sources (RMSE/MAE down, R² up)."),
        ("code", "pred = agent.predict(panel[panel.source=='urbanev'].tail(3000))\n"
                 "pred[['utilization_pred','expected_load','congestion_probability']].describe().round(3)"),
    ],
    "04_tariff_agent.ipynb": [
        ("md", "# 04 · Price Elasticity + Tariff Pricing Agent\n\n"
               "Elasticity is estimated from UrbanEV dynamic-pricing grids (association, "
               "**not** causal). The decisive finding is that elasticity is **state-dependent**: "
               "demand is near-inelastic when stations are busy (ε≈0) and elastic when idle "
               "(ε≈−0.5). This is *why* dynamic pricing beats flat — surge the inelastic peaks, "
               "discount the elastic off-peak."),
        ("code", PATH_SETUP + "from src.elasticity import estimate_elasticity, estimate_elasticity_by_state\n"
                 "print('overall:', round(estimate_elasticity()['elasticity'],3))\n"
                 "state = estimate_elasticity_by_state(); state['by_state']"),
        ("md", "A single averaged elasticity hides this and (by Jensen's inequality) makes "
               "dynamic pricing *lose* to flat. State-dependent elasticity fixes it.\n\n"
               "**Two operating points, reported honestly:**\n"
               "- *Revenue-neutral* — same average tariff as the ₹15 flat rate; the gain is "
               "pure price discrimination.\n"
               "- *Full learned* — also corrects the underpriced flat level (average price rises)."),
        ("code", "import pandas as pd\nfrom src.data import load_unified\nfrom src.agents.demand import DemandAgent\n"
                 "from src.agents.tariff import TariffAgent\nfrom src.elasticity import make_eps_fn\n"
                 "panel = load_unified(); ag = DemandAgent(); ag.fit_eval(panel)\n"
                 "pred = ag.predictions[ag.predictions.source=='urbanev'].merge(\n"
                 "    panel[['timestamp','location_id','energy_kwh']], on=['timestamp','location_id'])\n"
                 "fn = make_eps_fn(state['by_state'])\n"
                 "neutral = TariffAgent(2.0, 1.5, eps_fn=fn, revenue_neutral=True)\n"
                 "full = TariffAgent(2.0, 1.5, eps_fn=fn)\n"
                 "_, kN = neutral.simulate(pred); _, kF = full.simulate(pred)\n"
                 "pd.DataFrame({'revenue_neutral':kN,'full_learned':kF}).loc[\n"
                 "    ['revenue_gain_pct','offpeak_uplift_pct','avg_price_multiplier']]"),
    ],
    "05_monitor_eval.ipynb": [
        ("md", "# 05 · Monitoring & Learning Agent (feedback loop)\n\n"
               "The Monitor scores each pricing decision (revenue, off-peak uplift, "
               "wait-time reduction, pricing efficiency) and tunes the Tariff Agent's "
               "surge/discount sensitivity over episodes. The loop **discovers** that "
               "blanket off-peak discounts are revenue-negative under inelastic demand, "
               "converging on a surge-led policy."),
        ("code", PATH_SETUP + "from src.run_pipeline import main\nmain()"),
        ("md", "### Learning curve — composite objective rises as the agent learns"),
        ("code", "import pandas as pd, matplotlib.pyplot as plt\n"
                 "log = pd.read_csv('../outputs/monitor_episodes.csv')\n"
                 "ax = log.plot(x='episode_idx', y=['composite','wait_reduction_pct','revenue_gain_pct'], marker='o')\n"
                 "ax.set_title('Self-improvement across episodes'); ax.grid(alpha=.3); plt.show()\nlog.round(2)"),
        ("md", "### Consolidated headline KPIs"),
        ("code", "pd.read_csv('../outputs/final_kpis.csv').T"),
    ],
}


def build():
    for name, cells in NB.items():
        nb = new_notebook()
        nb.cells = [new_markdown_cell(c) if t == "md" else new_code_cell(c) for t, c in cells]
        nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}}
        with open(f"notebooks/{name}", "w") as f:
            nbf.write(nb, f)
        print("wrote notebooks/" + name)


if __name__ == "__main__":
    build()
