from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd
import yaml

from utils import META_DIR, PROJECT_ROOT, QLIB_DIR, ensure_project_dirs, setup_logger


CONFIG_PATH = PROJECT_ROOT / "configs" / "workflow_config_lightgbm_Alpha158_cn_custom.yaml"


def infer_segments() -> dict:
    cal_path = QLIB_DIR / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError("Missing Qlib calendar. Run 07_dump_qlib_bin.py first.")
    dates = pd.read_csv(cal_path, header=None)[0].astype(str).tolist()
    if len(dates) < 60:
        raise RuntimeError("Not enough Qlib calendar dates for Alpha158 workflow.")
    if len(dates) >= 1200:
        n_train = int(len(dates) * 0.60)
        n_valid = int(len(dates) * 0.20)
    else:
        n_train = int(len(dates) * 0.70)
        n_valid = int(len(dates) * 0.15)
    test_end = dates[-2] if len(dates) > n_train + n_valid + 1 else dates[-1]
    train = [dates[0], dates[n_train - 1]]
    valid = [dates[n_train], dates[n_train + n_valid - 1]]
    test = [dates[n_train + n_valid], test_end]
    return {"train": train, "valid": valid, "test": test, "all": [dates[0], dates[-1]]}


def build_config(segments: dict, deal_price: str) -> dict:
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
        },
        "backtest": {
            "start_time": segments["test"][0],
            "end_time": segments["test"][1],
            "account": 100000000,
            "benchmark": "SH000001",
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": deal_price,
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }
    return {
        "sys": {"path": []},
        "qlib_init": {"provider_uri": "data/qlib/cn_data", "region": "cn"},
        "market": "all",
        "benchmark": "SH000001",
        "data_handler_config": {
            "start_time": segments["all"][0],
            "end_time": segments["all"][1],
            "fit_start_time": segments["train"][0],
            "fit_end_time": segments["train"][1],
            "instruments": "all",
        },
        "port_analysis_config": port_analysis_config,
        "task": {
            "model": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.2,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                },
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": {
                            "start_time": segments["all"][0],
                            "end_time": segments["all"][1],
                            "fit_start_time": segments["train"][0],
                            "fit_end_time": segments["train"][1],
                            "instruments": "all",
                        },
                    },
                    "segments": {"train": segments["train"], "valid": segments["valid"], "test": segments["test"]},
                },
            },
            "record": [
                {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp", "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}},
                {"class": "SigAnaRecord", "module_path": "qlib.workflow.record_temp", "kwargs": {"ana_long_short": False, "ann_scaler": 252}},
                {"class": "PortAnaRecord", "module_path": "qlib.workflow.record_temp", "kwargs": {"config": port_analysis_config}},
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deal-price", choices=["open", "close"], default="open")
    args = parser.parse_args()
    ensure_project_dirs()
    logger = setup_logger("run_qlib", "08_run_alpha158_lightgbm.log")
    segments = infer_segments()
    config = build_config(segments, args.deal_price)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    pd.DataFrame(
        [{"segment": k, "start_date": v[0], "end_date": v[1]} for k, v in segments.items() if k != "all"]
    ).to_csv(META_DIR / "qlib_segments.csv", index=False, encoding="utf-8-sig")

    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    local_ext = list((PROJECT_ROOT / "qlib" / "qlib" / "data" / "_libs").glob("rolling*.pyd"))
    if local_ext:
        env["PYTHONPATH"] = str(PROJECT_ROOT / "qlib") + os.pathsep + env.get("PYTHONPATH", "")
    check = subprocess.run(
        [sys.executable, "-c", "import qlib.cli.run; import lightgbm"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "Qlib workflow dependencies are not ready. Install requirements with "
            "`python -m pip install -r requirements.txt`, or compile/install the local qlib source. "
            f"Original error: {check.stderr.strip() or check.stdout.strip()}"
        )
    logger.info("Running Qlib workflow by code with deal_price=%s", args.deal_price)
    import qlib
    from qlib.utils import flatten_dict, init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

    qlib.init(
        provider_uri=str(QLIB_DIR),
        region="cn",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "file:" + str((PROJECT_ROOT / "mlruns").resolve()),
                "default_exp_name": "sina_alpha158_lightgbm",
            },
        },
    )
    model = init_instance_by_config(config["task"]["model"])
    dataset = init_instance_by_config(config["task"]["dataset"])
    port_config = config["port_analysis_config"]
    port_config["strategy"]["kwargs"]["signal"] = (model, dataset)
    with R.start(experiment_name="sina_alpha158_lightgbm"):
        R.log_params(**flatten_dict(config["task"]))
        logger.info("Fitting LightGBM model")
        model.fit(dataset)
        recorder = R.get_recorder()
        R.save_objects(**{"params.pkl": model})
        logger.info("Generating predictions")
        SignalRecord(model, dataset, recorder).generate()
        logger.info("Skipping Qlib SigAnaRecord; IC is computed by scripts/09_analyze_results.py")
        logger.info("Generating portfolio analysis")
        PortAnaRecord(recorder, port_config, "day").generate()
    logger.info("Qlib workflow complete")


if __name__ == "__main__":
    main()
