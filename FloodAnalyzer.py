import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.stats import genpareto
from hdbscan import HDBSCAN
from typing import TypedDict, Dict, Set, Literal, Union, cast, Any
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, precision_score
from os import environ
from phd_code.infra.paths import FLOOD_SCORES_CSV
from pathlib import Path
from phd_code.infra.plot_style import FONT_SIZES, COLOURS, set_style, save_plot
from matplotlib.ticker import MaxNLocator
import os

set_style()
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "0"


warnings.filterwarnings("ignore", category=FutureWarning)

# KMeans memory leak
# environ["OMP_NUM_THREADS"] = '4'  # noqa
from sklearn.cluster import KMeans  # noqa


# --- TYPINGS ---

class ClusterStats(TypedDict):
    mean: float
    max: float
    std: float
    size: int
    is_extreme: bool
    rank: int


class ConfigResult(TypedDict):
    data: pd.DataFrame
    cluster_stats: Dict[int, ClusterStats]
    extreme_clusters: Set[int]
    cluster_model: KMeans | DBSCAN | HDBSCAN
    pretrained: bool


class BaseConfig(TypedDict, total=False):
    model: KMeans | DBSCAN | HDBSCAN


class KMeansConfig(BaseConfig):
    method: Literal["kmeans"]
    n_clusters: int


class DBSCANConfig(BaseConfig):
    method: Literal["dbscan"]
    eps: float
    min_samples: int


class RequiredHDBSCANConfig(BaseConfig):
    method: Literal["hdbscan"]
    min_cluster_size: int
    min_samples: int


class OptionalHDBSCANConfig(BaseConfig, total=False):
    cluster_selection_epsilon: float


class HDBSCANConfig(RequiredHDBSCANConfig, OptionalHDBSCANConfig):
    pass


ClusteringConfig = Union[KMeansConfig, DBSCANConfig, HDBSCANConfig]


# --- CLASSES ---


class ClusterFloodSignals:
    def __init__(self,
                 data: pd.DataFrame,
                 ts_col: str = "water_level",
                 percentile=95,
                 threshold=None
                 ):
        self._required_columns(data)
        self.ts_col = ts_col
        self.passed_threshold = threshold

        self.data = data.copy(deep=True)
        self.config_results: Dict[str, ConfigResult] = {}
        self.percentile = percentile
        self._gpd()

    # --- PROPERTIES ---

    @property
    def flood_months(self):
        return sorted(self.data["month_str"].unique())

    @property
    def config_names(self):
        return list(self.config_results.keys())

    @property
    def threshold(self):
        if self.passed_threshold:
            return self.passed_threshold
        else:
            return np.percentile(self.data[self.ts_col], self.percentile).item()

    # --- CLUSTERING ---

    def run(self, configs: list[ClusteringConfig],):
        x = self.data[self.ts_col].to_numpy().reshape(-1, 1)

        for config in configs:
            name = self._config_name(config)
            clusters, model = self._cluster(x, config)

            result = self.data.copy()
            result["cluster"] = clusters
            result["config"] = name

            cluster_stats = {}
            extreme_clusters = set()

            for cl in np.unique(clusters):
                cl_values: pd.Series = result[result["cluster"]
                                              == cl][self.ts_col]

                # Common stats for all clusters
                cluster_stats[cl] = {
                    "mean": cl_values.mean(),
                    "min": cl_values.min(),
                    "max": cl_values.max(),
                    "std": cl_values.std(),
                    "size": cl_values.size,
                }

                # Special handling for noise
                if cl == -1:
                    high_noise = cl_values[cl_values >= self.threshold]
                    cluster_stats[cl]["high_noise_count"] = len(high_noise)
                    cluster_stats[cl]["is_extreme"] = len(high_noise) > 0

                    if len(high_noise) > 0:
                        extreme_clusters.add(-1)
                else:
                    # Regular cluster
                    is_extreme = cluster_stats[cl]["mean"] >= self.threshold
                    cluster_stats[cl]["is_extreme"] = is_extreme

                    if is_extreme:
                        extreme_clusters.add(cl)

            ranked = sorted(cluster_stats.items(),
                            key=lambda item: -item[1]["mean"])
            for rank, (cl, _) in enumerate(ranked, start=1):
                cluster_stats[cl]["rank"] = rank

            # Mark extremes
            result["is_extreme"] = result["cluster"].isin(extreme_clusters)

            # Override: for noise cluster, only mark HIGH values as extreme
            if -1 in extreme_clusters:
                noise_mask = result["cluster"] == -1
                result.loc[noise_mask, "is_extreme"] = result.loc[noise_mask,
                                                                  self.ts_col] >= self.threshold

            self.config_results[name] = {
                "data": result,
                "cluster_stats": cluster_stats,
                "extreme_clusters": extreme_clusters,
                "cluster_model": model,
                "pretrained": config.get("model") is not None
            }

    def _config_name(self, config: ClusteringConfig):
        method = config["method"]

        part = ""
        if method == "kmeans":
            part = f"k{config["n_clusters"]}"  # type:ignore

        elif method == "dbscan":
            part = (
                f"eps{config["eps"]}_ms{config["min_samples"]}"  # type: ignore
            )

        elif method == "hdbscan":
            eps = f"eps{config["cluster_selection_epsilon"]}" if config.get(  # type:ignore
                "cluster_selection_epsilon", None) else ""
            mcs = config["min_cluster_size"]  # type:ignore

            part = (
                f"mcs{mcs}_ms{config["min_samples"]}{eps}"  # type:ignore
            )

        name = f"{method}_{part}"

        if config.get("model") is not None:
            name += "_pretrained"

        return name

    def _cluster(self, x, config: ClusteringConfig):
        if config.get("model") is not None:
            return self._cluster_with_trained_model(x, config)

        method = config["method"]

        if method == "kmeans":
            cfg = cast(KMeansConfig, config)
            model = KMeans(n_clusters=cfg["n_clusters"],
                           random_state=42,
                           init="k-means++",
                           algorithm="lloyd",
                           n_init=50,
                           tol=0.0001
                           )
            clusters = model.fit_predict(x)
            centers = model.cluster_centers_.flatten()

        elif method == "dbscan":
            cfg = cast(DBSCANConfig, config)
            model = DBSCAN(eps=cfg["eps"],
                           min_samples=cfg["min_samples"])
            clusters = model.fit_predict(x)
            centers = None

        elif method == "hdbscan":
            cfg = cast(HDBSCANConfig, config)
            model = HDBSCAN(min_cluster_size=cfg["min_cluster_size"],
                            allow_single_cluster=True,
                            min_samples=cfg["min_samples"],
                            cluster_selection_epsilon=config.get("cluster_selection_epsilon",
                                                                 0.0))
            clusters = model.fit_predict(x)
            centers = None

        else:
            raise ValueError(f"Unsupported method: {method}")

        return clusters, model

    def _cluster_with_trained_model(self, x, config: ClusteringConfig):
        """Use an already fitted model instead of training a new one."""
        if config.get("model") is None:
            raise Exception

        model = config["model"]  # type: ignore
        method = config["method"]

        if method == "kmeans" and isinstance(model, KMeans):
            clusters = model.predict(x)

        elif method == "dbscan" and isinstance(model, DBSCAN):
            if not hasattr(model, "labels_"):
                raise ValueError(
                    "DBSCAN trained_model must already be fitted.")
            clusters = model.labels_

        elif method == "hdbscan" and isinstance(model, HDBSCAN):
            if not hasattr(model, "labels_"):
                raise ValueError(
                    "HDBSCAN trained_model must already be fitted.")
            clusters = model.labels_

        else:
            raise ValueError(
                f"Trained model type {type(model).__name__} does not match config method '{method}'"
            )

        return clusters, model

    # --- EXTREME VALUE ANALYSIS ---

    def _gpd(self):
        self.gpd_data = pd.DataFrame({
            "day_str": self.data["day_str"],
            "month_str": self.data["month_str"]
        })

        exceedances = self._exceedances()

        if exceedances.empty:
            print("Warning: No exceedances found above the threshold.")
            self.gpd_data["exceedance"] = 0
            self.gpd_data["tail_prob"] = np.nan
            self.gpd_data["exceedance_quantile"] = np.nan
            self.gpd_data["is_extreme"] = False
            return

        shape, loc, scale = genpareto.fit(exceedances, floc=0)

        self.gpd_data["exceedance"] = self.data[self.ts_col] - \
            self.threshold
        self.gpd_data["exceedance"] = self.gpd_data["exceedance"].clip(
            lower=0)  # Replace negative with 0

        self.gpd_data["tail_prob"] = self.gpd_data["exceedance"].apply(
            lambda x: 1 -  # type: ignore
            genpareto.cdf(x, shape, loc=0, scale=scale) if x > 0 else np.nan
        )

        self.gpd_data["exceedance_quantile"] = self.gpd_data["exceedance"].apply(  # type: ignore
            lambda x: genpareto.cdf(  # type: ignore
                x, shape, loc=0, scale=scale) if x > 0 else np.nan
        )

        self.gpd_data["tail_prob"] = self.gpd_data["tail_prob"].fillna(0)
        self.gpd_data["exceedance_quantile"] = self.gpd_data["exceedance_quantile"].fillna(
            0)

        self.gpd_data["is_extreme"] = False
        self.gpd_data.loc[exceedances.index, "is_extreme"] = True

    def _exceedances(self, run_length: int = 2):
        df = self.data.copy(deep=True)

        df["above_thresh"] = df[self.ts_col] > self.threshold

        df["day_date"] = pd.to_datetime(df["day_str"])
        df = df.sort_values("day_date")

        df["group"] = (df["above_thresh"] !=
                       df["above_thresh"].shift()).cumsum()
        groups = df[df["above_thresh"]].groupby("group")

        peak_idxs = groups[self.ts_col].idxmax()
        filtered_idxs = []
        last_date = None
        for idx in peak_idxs:
            current_date = pd.to_datetime(
                self.data.loc[idx, "day_str"])  # type: ignore
            if last_date is None or (current_date - last_date).days > run_length:
                filtered_idxs.append(idx)
                last_date = current_date

        exceedances = df.loc[filtered_idxs,
                             self.ts_col] - self.threshold
        return exceedances

    # --- INTERNAL UTILITIES ---

    def _required_columns(self, data: pd.DataFrame):
        col_names = {"day_str", "month_str"}
        assert col_names.issubset(
            data.columns), f"Missing required columns: {col_names - set(data.columns)}"


class ResultAnalyzer:
    def __init__(self, clustered_signals: ClusterFloodSignals):
        self._clustered_signals = clustered_signals
        self._active_config = None
        self.ts_col = clustered_signals.ts_col

    # --- PROPERTIES ---

    @property
    def active_config(self):
        if self._active_config is None:
            raise Exception("Select config first")
        return self._active_config

    @active_config.setter
    def active_config(self, config_name: str):
        if config_name in self._clustered_signals.config_results:
            self._active_config = config_name
        else:
            raise ValueError(f"Config '{config_name}' not found")

    @property
    def config_results(self) -> ConfigResult:
        return self._clustered_signals.config_results[self.active_config]

    @property
    def flood_months(self):
        return self._clustered_signals.flood_months

    @property
    def gpd_data(self):
        return self._clustered_signals.gpd_data

    @property
    def clustering_model(self):
        return self._clustered_signals.config_results[self.active_config]["cluster_model"]

    @property
    def config_cluster_stats(self):
        stats_dict = self.config_results["cluster_stats"]
        return (pd.DataFrame
                .from_dict(stats_dict, orient="index")
                .reset_index(names="Cluster")
                .assign(size_pct=lambda df: 100 * df["size"] / df["size"].sum())
                .round(2)
                )

    # --- RESULTS AND SUMMARIZATION ---

    def extreme_days_per_cluster(self, selected_months=None):
        # self.print_title("Days in Extreme Clusters", all_configs=True)

        monthly_summary = pd.DataFrame(index=self.flood_months)

        for name, result in self._clustered_signals.config_results.items():
            data = result["data"]
            month_counts = data[data["is_extreme"]
                                ].groupby("month_str").size()
            monthly_summary[name] = month_counts.reindex(
                self.flood_months, fill_value=0)
            monthly_summary.sort_index(inplace=True)

        if selected_months:
            monthly_summary = monthly_summary[monthly_summary.index.isin(
                selected_months)]

        return monthly_summary.astype(int)

    def gpd_extreme_days(self):
        gpd_data = self.gpd_data
        gpd_data["day_num"] = pd.to_datetime(
            gpd_data["day_str"]).dt.day

        gpd_extremes = gpd_data[gpd_data["is_extreme"]].copy()

        quantiles = gpd_extremes["exceedance_quantile"]
        q1 = quantiles.quantile(0.33)
        q2 = quantiles.quantile(0.66)

        def classify_severity(x):
            if x == 0:
                return "N/A"
            elif x <= q1:
                return "minor"
            elif x <= q2:
                return "moderate"
            else:
                return "major"

        gpd_extremes["severity"] = quantiles.apply(classify_severity)
        gpd_summary = (
            gpd_extremes.groupby(["month_str", "severity"])["day_num"]
            .apply(lambda days: ", ".join(str(day) for day in sorted(days)))
            .unstack(fill_value="")
            .reset_index()
        )

        gpd_summary.columns.name = None
        gpd_summary = gpd_summary.reindex(
            columns=["month_str", "major", "moderate", "minor"], fill_value="")
        gpd_summary = gpd_summary[["month_str", "major", "moderate", "minor"]]
        gpd_summary = gpd_summary.rename(columns={
            "minor": "gpd_minor_days",
            "moderate": "gpd_moderate_days",
            "major": "gpd_major_days",
        })

        gpd_summary.set_index("month_str", inplace=True)
        gpd_summary = gpd_summary.reindex(self.flood_months, fill_value="")

        return gpd_summary

    def cluster_extreme_days(self):
        all_days = pd.concat([
            self._clustered_signals.config_results[config]["data"][[
                "day_str", "month_str", "is_extreme", "config"]]
            for config in self._clustered_signals.config_results.keys()
        ])
        unique_extreme_days = all_days[all_days["is_extreme"]
                                       ].drop_duplicates(subset=["day_str"])

        return unique_extreme_days

    def flood_consensus(self, alpha=0.5, beta=0.5):

        extreme_clusters_days = self.extreme_days_per_cluster()
        total_extreme_days = extreme_clusters_days.sum(axis=1)
        n_configs = extreme_clusters_days.shape[1]

        # --- AGREEMENT SCORE: how many configs detected atleast one extreme day ---

        agreement_score = ((extreme_clusters_days > 0).sum(axis=1)) / n_configs

        # --- DAILY DENSITY: Normalize by number of days in the month ---

        all_days = pd.concat([
            self._clustered_signals.config_results[config]["data"][[
                "day_str", "month_str", "is_extreme", "config"]]
            for config in self._clustered_signals.config_results.keys()
        ])

        unique_extreme_days = all_days[all_days["is_extreme"]
                                       ].drop_duplicates(subset=["day_str"])

        days_per_month = all_days.groupby("month_str")["day_str"].nunique()
        extreme_days_per_month = unique_extreme_days.groupby("month_str"
                                                             )["day_str"].nunique()

        daily_density = (extreme_days_per_month / days_per_month)
        daily_density = daily_density.reindex(extreme_clusters_days.index,
                                              fill_value=0.0)

        # --- CONFIG DENSITY: average number of extreme days per config ---

        config_density = total_extreme_days / n_configs

        flood_score = (
            alpha * config_density.rank(pct=True) + beta * daily_density.rank(pct=True))

        # --- CATEGORICAL LABELLING ---
        categories = ["No Flood", "Low Agreement", "Marginal / Ambiguous",
                      "Flood Indicated", "Major Flood"]

        def categorize(row):
            intensity = row["intensity_score"]
            agreement = row["agreement_score"]
            flood_score = row["flood_score"]

            if intensity == 0:
                return "No Flood"
            if round((agreement * n_configs)) == intensity:
                return "Low Agreement"

            if flood_score >= 0.7:
                return "Major Flood"
            if flood_score >= 0.4:
                return "Flood Indicated"

            return "Marginal / Ambiguous"

        consensus_score = pd.DataFrame({
            "intensity_score": total_extreme_days,
            "agreement_score": agreement_score.round(3),
            "config_density": config_density.round(3),
            "daily_density": daily_density.round(3),
            "flood_score": flood_score.round(3)
        })
        consensus_score["category"] = consensus_score.apply(categorize, axis=1)
        consensus_score["category"] = consensus_score["category"].astype(
            pd.CategoricalDtype(categories=categories, ordered=True)
        )
        consensus_score = consensus_score.iloc[:,
                                               [-1, -2] + list(range(consensus_score.shape[1]-2))]
        consensus_score = consensus_score.reset_index(names="month_str")
        consensus_score.to_csv(FLOOD_SCORES_CSV, index=False)

        return consensus_score

    # --- METHODS FOR INDIVIDUAL CONFIGS

    def days_in_cluster(self, return_result=False):
        data = self.config_results["data"]

        self.print_title("Days in Clusters")
        return data.groupby(["month_str", "cluster"]).size().unstack(fill_value=0).sort_index()

    def gpd_vs_cluster_cm(self, return_result=False):
        config_data = self.config_results["data"]
        gpd_data = self.gpd_data

        self.print_title("GPD vs Cluster Extreme Classification")

        y_true = gpd_data['is_extreme'].astype(bool)
        y_pred = config_data['is_extreme'].astype(bool)

        cm = confusion_matrix(y_true, y_pred, labels=[True, False])
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if return_result:
            return cm, precision, cm

        print(f"Confusion Matrix\n{cm}")
        print("")

        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")

    def extreme_days(self, selected_months=None, full_month_str=False,
                     cluster_number: int | None = None):
        config_data = self.config_results["data"].copy()
        config_data["day_num"] = pd.to_datetime(
            config_data["day_str"]).dt.day

        cluster_extremes = config_data[config_data["is_extreme"]]
        if cluster_number:
            cluster_extremes = cluster_extremes.query(
                "cluster == @cluster_number")

        cluster_summary = (
            cluster_extremes
            .groupby(["month_str"])
            .agg(
                days=("day_num", lambda s: ", ".join(str(day) for day in s)),
                n_days=("day_num", "count")

            )
            # .assign(n_days= lambda df: df["n_days"].replace("", np.nan))
            .reset_index()
        )
        cluster_summary["n_days"] = pd.to_numeric(
            cluster_summary["n_days"], errors="coerce")

        gpd_extreme_days = self.gpd_extreme_days()

        extreme_days = cluster_summary
        extreme_days = (
            pd.merge(cluster_summary, gpd_extreme_days,
                     on="month_str", how="outer")
            .fillna("")
        )

        if selected_months:
            extreme_days = extreme_days[extreme_days["month_str"].isin(
                selected_months)]

        if full_month_str:
            extreme_days["month_str"] = (
                pd.to_datetime(extreme_days["month_str"])
                .dt.strftime("%B %Y")
            )
        extreme_days["n_days"].replace("", np.nan, inplace=True)

        return extreme_days

    def missing_months(self):
        data = self.config_results["data"]
        cluster_stats = self.config_results["cluster_stats"]

        self.print_title("Missing Months Report")

        for cluster in sorted(data["cluster"].unique()):
            in_cluster = data["cluster"] == cluster
            cluster_data = data[in_cluster]

            is_extreme = cluster_stats[cluster]["is_extreme"]
            rank = cluster_stats[cluster]["rank"]
            diff = set(self.flood_months) - set(cluster_data["month_str"])

            print(
                f"Cluster {cluster} {"(Extreme)" if is_extreme else ""}")
            print(
                f"\tMissing months ({len(diff)}): {", ".join(diff) if len(diff) > 0 else "N/A"}")
            print(f"\tRank: {rank}")
            print("")

    def compare_clustered_days(self, series_1: pd.Series, series_2: pd.Series):
        series_1 = series_1.copy()
        series_2 = series_2.copy()

    # --- PLOTS ---

    def plot_extreme_days_per_cluster(self, save=False, **kwargs):
        data = self.extreme_days_per_cluster(**kwargs)

        plt.figure(figsize=(10, max(6, 0.5 * len(data))))
        sns.heatmap(
            data,
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Days in Extreme Cluster', "shrink": 0.5},
            annot=True,
            fmt='d',
        )
        plt.ylabel("Flood Month")
        plt.xlabel("Clustering Config")
        plt.xticks(rotation=-10, ha="left")
        plt.yticks()
        plt.tight_layout()
        plt.show()

    def plot_days_in_cluster(self):
        data = self.days_in_cluster()

        plt.figure(figsize=(10, max(6, 0.4 * len(data))))
        sns.heatmap(
            data,
            cmap="YlOrRd",
            linewidths=0,
            annot=True,
            fmt='d',
            cbar_kws={'label': 'Number of Days'}
        )
        plt.title(f"Days in Cluster ({self.active_config})", fontsize=14)
        plt.xlabel("Cluster ID")
        plt.ylabel("Month")
        plt.tight_layout()
        plt.show()

    def plot_kde(self, ts_type="Magnitude"):
        data = self.config_results["data"]
        plt.figure(figsize=(5.5, 3))

        cluster_colors = self._get_cluster_colors()
        sorted_clusters = sorted(data["cluster"].unique())
        palette = [cluster_colors[cluster] for cluster in sorted_clusters]

        ax = sns.kdeplot(data=data, x=self.ts_col, hue="cluster",
                         fill=True, alpha=0.6, palette=palette)
        ax.set_xlabel(f"Dynamic Coastal Forcing Index")
        ax.set_ylabel("Density")
        ax.tick_params("both", labelsize=FONT_SIZES["TICK"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, color=COLOURS["TEXT_BROWN"],
                linewidth=0.2, alpha=0.2)
        ax.set_title("Distribution of Daily DCFI Values Per Cluster",
                     fontsize=FONT_SIZES["PLOT_TITLE"])

        legend = plt.gca().get_legend()
        legend.set_ncols(3)
        legend.set_title("Clusters")
        frame = legend.get_frame()
        frame.set_edgecolor("none")

        sns.despine(left=True)
        save_plot(f"KDE Distribution of Days Per Cluster")
        plt.show()

    def plot_gpd_vs_cluster_cm(self):
        result = self.gpd_vs_cluster_cm(return_result=True)

        if result is None:
            return

        cm, precision, recall = result

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Extreme", "Not Extreme"]
        )
        disp.plot(cmap="Blues", values_format='d')
        plt.title(f"GPD vs Cluster Confusion Matrix ()")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_flood_consensus(self, **kwargs):
        consensus = self.flood_consensus(**kwargs)
        palette = {
            "Major Flood": "#d62728",
            "Flood Indicated": "#ff7f0e",
            "Marginal / Ambiguous": "#1f77b4",
            "Low Agreement": "#d9ef8b",
            "No Flood": "#2ca02c"
        }

        plt.figure(figsize=(14, 6))

        barplot = sns.barplot(
            x="month_str",
            y="flood_score",
            hue="category",
            data=consensus,
            palette=palette,
            dodge=False
        )

        plt.xticks(rotation=-35, ha="left")
        plt.xlabel("Month")
        plt.ylabel("Flood Likelihood Score")
        plt.title("Flood Likelihood and Consensus Classification")
        plt.legend(title="Flood Category")
        plt.tight_layout()
        plt.show()

    # --- INTERNAL UTILITIES ---

    def list_configs(self):
        for key in self._clustered_signals.config_results.keys():
            print(key)

    def print_title(self, title, all_configs=False):
        if all_configs:
            title = f"{title} (all configs)"
        else:
            title = f"{title} ({self.active_config})"

        print("-" * len(title))
        print(title)
        print("-" * len(title))

        print("")

    def _get_cluster_colors(self):
        data = self.config_results["data"]

        unique_clusters = sorted(data["cluster"].unique())
        palette = COLOURS["GLOBAL_PALETTE"]
        return {cluster: color for cluster, color in zip(unique_clusters, palette)}

    def save_fig(self, filename: str):
        base_path = Path("C:/Users/steve/Desktop/")
        plt.savefig(base_path / filename, dpi=1000, bbox_inches="tight")
