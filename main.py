from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_capstone.artifacts import (
    create_run_artifacts,
    load_run_artifacts,
    write_artifact_manifest,
    write_latest_run,
)
from my_capstone.config import RunConfig, load_run_config
from my_capstone.pipeline import (
    execute_deepcase_predict,
    execute_deepcase_prepare,
    execute_deepcase_run,
    execute_deepcase_train,
)
from my_capstone.utils import capture_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="My_Capstone 專題的 DeepCASE 執行入口。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--config",
            required=True,
            help="JSON 設定檔路徑。",
        )

    prepare_parser = subparsers.add_parser(
        "deepcase-prepare",
        help="驗證或建立可直接給 DeepCASE 使用的 processed CSV。",
    )
    add_config_argument(prepare_parser)
    prepare_parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="即使既有 processed CSV 看起來有效，也強制重新建立。",
    )

    train_parser = subparsers.add_parser(
        "deepcase-train",
        help="訓練 DeepCASE，並保存可重用的訓練產物。",
    )
    add_config_argument(train_parser)
    train_parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="在訓練前強制重建 processed CSV。",
    )

    predict_parser = subparsers.add_parser(
        "deepcase-predict",
        help="載入既有 DeepCASE run，重新產生預測結果。",
    )
    add_config_argument(predict_parser)
    predict_parser.add_argument(
        "--load-run",
        required=True,
        help="要載入的 run ID；填入 latest 可重用最近一次保存的訓練成果。",
    )
    predict_parser.add_argument(
        "--input-csv",
        help=(
            "可選的 DeepCASE-ready CSV。"
            "若省略，會直接重用來源 run 保存的 test split。"
        ),
    )

    run_parser = subparsers.add_parser(
        "deepcase-run",
        help="一次完成資料準備、DeepCASE 訓練、預測與所有輸出保存。",
    )
    add_config_argument(run_parser)
    run_parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="在完整流程前強制重建 processed CSV。",
    )

    return parser


def command_mode(command: str) -> str:
    mapping = {
        "deepcase-train": "train",
        "deepcase-predict": "predict",
        "deepcase-run": "full",
    }
    return mapping[command]


def run_with_artifacts(
    command: str,
    config: RunConfig,
    *,
    force_prepare: bool = False,
    load_run: str | None = None,
    input_csv: str | None = None,
) -> int:
    mode_name = command_mode(command)
    artifacts = create_run_artifacts(
        results_root=config.results_root,
        dataset_name=config.dataset_name,
        device_name=config.resolved_device,
        mode_name=mode_name,
    )

    source_artifacts = None
    if load_run is not None:
        source_artifacts = load_run_artifacts(config.results_root, load_run)

    with capture_output(artifacts.stdout_log):
        print(f"[My_Capstone] 指令={command}")
        print(f"[My_Capstone] run_id={artifacts.run_id}")
        print(f"[My_Capstone] 裝置={config.resolved_device}")
        print(f"[My_Capstone] 輸出目錄={artifacts.run_dir}")

        if command == "deepcase-train":
            summary = execute_deepcase_train(
                config=config,
                artifacts=artifacts,
                force_prepare=force_prepare,
            )
        elif command == "deepcase-predict":
            if source_artifacts is None:
                raise ValueError("--load-run is required for deepcase-predict.")
            summary = execute_deepcase_predict(
                config=config,
                artifacts=artifacts,
                source_artifacts=source_artifacts,
                input_csv=input_csv,
            )
        else:
            summary = execute_deepcase_run(
                config=config,
                artifacts=artifacts,
                force_prepare=force_prepare,
            )

    write_artifact_manifest(
        artifacts=artifacts,
        command=command,
        config_path=config.config_path,
        source_run_id=source_artifacts.run_id if source_artifacts else None,
        summary=summary,
    )
    write_latest_run(config.results_root, artifacts.run_id)
    print(f"[My_Capstone] completed run_id={artifacts.run_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_run_config(args.config)

    if args.command == "deepcase-prepare":
        result = execute_deepcase_prepare(
            config=config,
            force_prepare=args.force_prepare,
        )
        print(
            f"[My_Capstone] processed_csv={result.output_path} 是否重用既有檔案={result.reused_existing}"
        )
        return 0

    if args.command == "deepcase-train":
        return run_with_artifacts(
            args.command,
            config,
            force_prepare=args.force_prepare,
        )

    if args.command == "deepcase-predict":
        return run_with_artifacts(
            args.command,
            config,
            load_run=args.load_run,
            input_csv=args.input_csv,
        )

    return run_with_artifacts(
        args.command,
        config,
        force_prepare=args.force_prepare,
    )


if __name__ == "__main__":
    raise SystemExit(main())
