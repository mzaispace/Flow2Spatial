
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flow2Spatial 完整复现环境与数据审计脚本

检查内容：
1. Python / PyTorch / CUDA / GPU 环境
2. Git 分支、提交版本、工作区状态
3. Flow2Spatial 本地源码导入路径
4. Generator / Model 各函数的真实 API
5. 大鼠肠道 CSV 数据
6. Mask 和 Hcoordinate.pkl 内部结构
7. Gut_reference.h5ad 和作者重建结果
8. 小脑 MAGIC-seq / MALDI-MSI / PLATO 数据
9. 输出完整审计报告

本脚本只读取数据，不修改原始数据，不执行模型训练。
"""

import sys
import os
import json
import pickle
import inspect
import importlib
import subprocess
import traceback

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 项目路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
DATASETS = ROOT / "datasets"
REPORT_DIR = ROOT / "experiments"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def title(text):
    print("\n" + "=" * 75)
    print(f" {text}")
    print("=" * 75)


def info(key, value):
    print(f"[INFO] {key}: {value}")


def warning(message):
    print(f"[WARNING] {message}")


def error(message):
    print(f"[ERROR] {message}")


def run_command(command):
    """执行系统命令并返回输出。"""
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False
        )

        return result.stdout.strip()

    except Exception as exc:
        return str(exc)


# ============================================================
# 2. Git 仓库检查
# ============================================================

def inspect_git():
    title("GIT REPOSITORY")

    commands = {
        "Branch": ["git", "branch", "--show-current"],
        "Commit": ["git", "rev-parse", "HEAD"],
        "Status": ["git", "status", "--short"],
        "Remotes": ["git", "remote", "-v"],
    }

    for name, command in commands.items():
        result = run_command(command)
        info(name, result or "(empty)")

    branch = run_command(
        ["git", "branch", "--show-current"]
    )

    if branch != "repro":
        warning(
            f"当前分支为 {branch}，"
            "建议在自己的 repro 分支开展源码修改。"
        )


# ============================================================
# 3. Python / CUDA 环境检查
# ============================================================

def inspect_environment():
    title("PYTHON / CUDA ENVIRONMENT")

    info("Python", sys.version)
    info("Executable", sys.executable)
    info("Project root", ROOT)

    import torch
    import torchvision
    import scipy
    import anndata
    import shapely
    import cvxpy

    info("PyTorch", torch.__version__)
    info("TorchVision", torchvision.__version__)
    info("NumPy", np.__version__)
    info("Pandas", pd.__version__)
    info("SciPy", scipy.__version__)
    info("AnnData", anndata.__version__)
    info("Shapely", shapely.__version__)
    info("CVXPY", cvxpy.__version__)

    info("CUDA available", torch.cuda.is_available())
    info("Torch CUDA runtime", torch.version.cuda)

    if torch.cuda.is_available():

        info("GPU count", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):

            info(
                f"GPU {i}",
                torch.cuda.get_device_name(i)
            )

            info(
                f"GPU {i} capability",
                torch.cuda.get_device_capability(i)
            )

        # 实际执行一次小型 CUDA 运算
        try:
            x = torch.randn(
                1024,
                1024,
                device="cuda"
            )

            y = x @ x

            torch.cuda.synchronize()

            info(
                "CUDA matrix multiplication",
                f"SUCCESS, mean={y.mean().item():.6f}"
            )

            info(
                "GPU allocated memory",
                f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )

            del x, y
            torch.cuda.empty_cache()

        except Exception as exc:
            error(f"CUDA computation failed: {exc}")

    else:
        warning("PyTorch 没有识别 CUDA GPU。")


# ============================================================
# 4. Flow2Spatial 源码和 API 检查
# ============================================================

def import_submodule(name):
    """
    显式导入子模块。

    不依赖 Flow2Spatial.__init__.py
    是否导出了 generator / model。
    """

    try:
        return importlib.import_module(
            f"Flow2Spatial.{name}"
        )

    except Exception as exc:
        error(f"Cannot import Flow2Spatial.{name}: {exc}")
        traceback.print_exc()
        return None


def inspect_api():
    title("FLOW2SPATIAL SOURCE AND API")

    import Flow2Spatial

    info(
        "Flow2Spatial source",
        Flow2Spatial.__file__
    )

    # 检查实际是否来自本地 Fork 项目
    try:
        source_path = Path(
            Flow2Spatial.__file__
        ).resolve()

        source_path.relative_to(ROOT.resolve())

        info("Local source verification", "PASS")

    except ValueError:
        warning(
            "Flow2Spatial 当前导入位置不在本地项目目录中，"
            "请检查 editable install。"
        )

    # 显式导入子模块
    generator = import_submodule("generator")
    model = import_submodule("model")

    if generator:
        info("Generator module", generator.__file__)

    if model:
        info("Model module", model.__file__)

    print("\n---------- Generator API ----------")

    generator_functions = [
        "omics",
        "histology",
        "random"
    ]

    for name in generator_functions:

        if generator is None:
            break

        func = getattr(generator, name, None)

        if func is None:
            error(f"Generator.{name} not found")
            continue

        try:
            info(
                f"generator.{name}",
                inspect.signature(func)
            )

            info(
                f"source file of {name}",
                inspect.getsourcefile(func)
            )

        except Exception as exc:
            warning(str(exc))

    print("\n---------- Model API ----------")

    model_functions = [
        "preparation",
        "training",
        "reconstruction"
    ]

    for name in model_functions:

        if model is None:
            break

        func = getattr(model, name, None)

        if func is None:
            error(f"Model.{name} not found")
            continue

        try:
            info(
                f"model.{name}",
                inspect.signature(func)
            )

            info(
                f"source file of {name}",
                inspect.getsourcefile(func)
            )

        except Exception as exc:
            warning(str(exc))

    return generator, model


# ============================================================
# 5. 通用对象结构检查
# ============================================================

def inspect_object(name, obj, depth=0, max_depth=3):
    """递归检查 pickle 对象结构，不进行数据修改。"""

    prefix = "  " * depth

    if isinstance(obj, dict):

        print(
            f"{prefix}{name}: dict, "
            f"length={len(obj)}, "
            f"keys={list(obj.keys())[:20]}"
        )

        if depth < max_depth:

            for key, value in list(obj.items())[:5]:

                inspect_object(
                    str(key),
                    value,
                    depth + 1,
                    max_depth
                )

    elif isinstance(obj, (list, tuple)):

        print(
            f"{prefix}{name}: {type(obj).__name__}, "
            f"length={len(obj)}"
        )

        if depth < max_depth:

            for i, value in enumerate(obj[:3]):

                inspect_object(
                    f"item_{i}",
                    value,
                    depth + 1,
                    max_depth
                )

    elif isinstance(obj, np.ndarray):

        print(
            f"{prefix}{name}: ndarray, "
            f"shape={obj.shape}, "
            f"dtype={obj.dtype}"
        )

        if obj.size > 0:

            print(
                f"{prefix}preview: "
                f"{obj.reshape(-1)[:5]}"
            )

            if obj.dtype == bool:
                print(
                    f"{prefix}true count: "
                    f"{int(obj.sum())}"
                )

    elif isinstance(obj, pd.DataFrame):

        print(
            f"{prefix}{name}: DataFrame, "
            f"shape={obj.shape}"
        )

        print(
            f"{prefix}columns: "
            f"{obj.columns.tolist()}"
        )

        print(obj.head(3).to_string(index=False))

    elif isinstance(obj, pd.Series):

        print(
            f"{prefix}{name}: Series, "
            f"shape={obj.shape}, dtype={obj.dtype}"
        )

    else:

        print(
            f"{prefix}{name}: "
            f"{type(obj).__name__}, "
            f"value={repr(obj)[:200]}"
        )


# ============================================================
# 6. Pickle 文件检查
# ============================================================

def inspect_pickle(path):

    title(f"PICKLE DATA: {path.name}")

    if not path.exists():
        error(f"File missing: {path}")
        return

    info("File size", f"{path.stat().st_size / 1024:.2f} KB")

    try:

        # 仅对来自可信官方仓库的 pickle 文件使用
        with open(path, "rb") as f:
            obj = pickle.load(f)

        inspect_object("root", obj)

    except Exception as exc:
        error(f"Pickle loading failed: {exc}")
        traceback.print_exc()


# ============================================================
# 7. CSV 文件检查
# ============================================================

def inspect_csv(path):

    title(f"CSV DATA: {path.name}")

    if not path.exists():
        error(f"File missing: {path}")
        return

    df = pd.read_csv(path)

    info("Shape", df.shape)
    info("Columns", df.columns.tolist())

    print("\n---------- Missing Values ----------")

    print(
        df.isna().sum()[
            df.isna().sum() > 0
            ].to_string()
    )

    print("\n---------- Data Preview ----------")

    print(df.head(5).to_string(index=False))

    important_columns = [
        "PG.Genes",
        "PG.ProteinAccessions"
    ]

    for col in important_columns:

        if col in df.columns:

            info(
                f"{col} unique count",
                df[col].nunique(dropna=True)
            )


# ============================================================
# 8. AnnData 文件检查
# ============================================================

def inspect_h5ad(path):

    title(f"ANNDATA: {path.name}")

    if not path.exists():
        error(f"File missing: {path}")
        return

    from anndata import read_h5ad

    adata = None

    try:

        # 使用 backed 模式，只读取元数据
        adata = read_h5ad(
            path,
            backed="r"
        )

        info("Shape", adata.shape)
        info("Observation count", adata.n_obs)
        info("Variable count", adata.n_vars)

        info(
            "Observation columns",
            adata.obs.columns.tolist()
        )

        info(
            "Variable columns",
            adata.var.columns.tolist()
        )

        info(
            "Obsm keys",
            list(adata.obsm.keys())
        )

        info(
            "Layers",
            list(adata.layers.keys())
        )

        info(
            "Raw available",
            adata.raw is not None
        )

        if adata.raw is not None:
            info(
                "Raw shape",
                adata.raw.shape
            )

        print("\n---------- Spatial Coordinates ----------")

        for key in [
            "spatial",
            "X_spatial",
            "spatial_original"
        ]:

            if key in adata.obsm:

                coords = np.asarray(
                    adata.obsm[key]
                )

                info(
                    f"obsm[{key}] shape",
                    coords.shape
                )

                print(
                    f"Preview:\n{coords[:3]}"
                )

        print("\n---------- Cluster Information ----------")

        if "cluster" in adata.obs:

            print(
                adata.obs["cluster"]
                .value_counts()
                .head(20)
                .to_string()
            )

        else:
            warning("No 'cluster' column in adata.obs")

        print("\n---------- Variable Name Preview ----------")

        print(
            list(adata.var_names[:10])
        )

    except Exception as exc:

        error(f"AnnData inspection failed: {exc}")
        traceback.print_exc()

    finally:

        if adata is not None:

            try:
                adata.file.close()
            except Exception:
                pass


# ============================================================
# 9. 文件清单检查
# ============================================================

def inspect_file_inventory():

    title("OFFICIAL DATA FILE INVENTORY")

    targets = [
        TESTS / "Gut_reference.h5ad",
        TESTS / "Hcoordinate.pkl",
        TESTS / "Recontruct_weights_gut.pkl",
        TESTS / "adata.h5ad",
        TESTS / "df_pro_gut.csv",
        TESTS / "mask",

        DATASETS / "Cerebellum-MAGIC-seq.h5ad",
        DATASETS / "Cerebellum-MALDI-MSI.h5ad",
        DATASETS / "Cerebellum-PLATO.h5ad",
        ]

    for path in targets:

        if path.exists():

            info(
                str(path.relative_to(ROOT)),
                f"FOUND ({path.stat().st_size / 1024**2:.2f} MB)"
            )

        else:
            warning(f"MISSING: {path.relative_to(ROOT)}")


# ============================================================
# 10. 主程序
# ============================================================

def main():

    # 创建报告文件
    report_file = REPORT_DIR / "data_audit.txt"

    import contextlib

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    with open(
            report_file,
            "w",
            encoding="utf-8"
    ) as report:

        original_stdout = sys.stdout

        sys.stdout = Tee(
            original_stdout,
            report
        )

        try:

            title("FLOW2SPATIAL REPRODUCTION AUDIT")

            info("Project", ROOT)

            inspect_git()

            inspect_environment()

            inspect_file_inventory()

            inspect_api()

            # 肠道真实蛋白质数据
            inspect_csv(
                TESTS / "df_pro_gut.csv"
            )

            # Tissue Mask
            inspect_pickle(
                TESTS / "mask"
            )

            # Histology 原始坐标
            inspect_pickle(
                TESTS / "Hcoordinate.pkl"
            )

            # 原始权重仅检查文件大小，不直接反序列化
            weight_path = (
                    TESTS / "Recontruct_weights_gut.pkl"
            )

            title("PRETRAINED MODEL WEIGHTS")

            if weight_path.exists():

                info(
                    "Weight file",
                    weight_path
                )

                info(
                    "File size",
                    f"{weight_path.stat().st_size / 1024**2:.2f} MB"
                )

            else:
                error("Pretrained weights missing")

            # 肠道空间组学
            inspect_h5ad(
                TESTS / "Gut_reference.h5ad"
            )

            # 作者提供的重建结果
            inspect_h5ad(
                TESTS / "adata.h5ad"
            )

            # 小脑多组学
            inspect_h5ad(
                DATASETS / "Cerebellum-MAGIC-seq.h5ad"
            )

            inspect_h5ad(
                DATASETS / "Cerebellum-MALDI-MSI.h5ad"
            )

            inspect_h5ad(
                DATASETS / "Cerebellum-PLATO.h5ad"
            )

            title("AUDIT COMPLETED")

            info("Report", report_file)

            print("\n请将 data_audit.txt 提供给后续复现分析。")

        finally:

            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
