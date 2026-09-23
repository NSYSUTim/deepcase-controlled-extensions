from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my_capstone.utils import capture_output, save_json
from my_capstone_research.common import (
    create_research_artifacts,
    load_research_artifacts,
    write_latest_research_run,
    write_research_manifest,
)
from my_capstone_research.registry import get_research_pipeline, load_research_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="My_Capstone 研究版 DeepCASE 入口。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--config",
            required=True,
            help="研究方法的 JSON 設定檔。",
        )

    prepare_parser = subparsers.add_parser("prepare", help="建立研究方法需要的前處理規格說明。")
    add_config_argument(prepare_parser)

    train_parser = subparsers.add_parser("train", help="訓練研究方法並保存 builder / interpreter。")
    add_config_argument(train_parser)

    predict_parser = subparsers.add_parser("predict", help="載入既有 research run，直接做預測。")
    add_config_argument(predict_parser)
    predict_parser.add_argument(
        "--load-run",
        required=True,
        help="要載入的 research run ID；可用 latest。",
    )
    predict_parser.add_argument(
        "--input-csv",
        help="可選：指定新的 processed CSV。未提供時沿用來源 run 的 test_sequences.save。",
    )

    run_parser = subparsers.add_parser("run", help="完整執行研究方法：train + predict。")
    add_config_argument(run_parser)

    return parser


def run_command(
    command: str,
    config_path: str,
    *,
    load_run: str | None = None,
    input_csv: str | None = None,
) -> int:
    config = load_research_config(config_path)
    pipeline = get_research_pipeline(config.method_name)
    artifacts = create_research_artifacts(
        results_root=config.results_root,
        method_name=config.method_name,
        variant_name=config.variant_name,
    )

    source_artifacts = None
    if load_run is not None:
        source_artifacts = load_research_artifacts(
            results_root=config.results_root,
            method_name=config.method_name,
            run_id=load_run,
        )

    with capture_output(artifacts.stdout_log):
        print(f"[My_Capstone Research] command={command}")
        print(f"[My_Capstone Research] method={config.method_name}")
        print(f"[My_Capstone Research] run_id={artifacts.run_id}")
        print(f"[My_Capstone Research] device={config.resolved_device}")
        print(f"[My_Capstone Research] run_dir={artifacts.run_dir}")

        save_json(artifacts.research_config_json, config.as_dict())

        if command == "prepare":
            generated = pipeline.execute_prepare(
                config=config,
                artifacts=artifacts,
                source_artifacts=source_artifacts,
                input_csv=input_csv,
            )
        elif command == "train":
            generated = pipeline.execute_train(
                config=config,
                artifacts=artifacts,
                source_artifacts=source_artifacts,
                input_csv=input_csv,
            )
        elif command == "predict":
            generated = pipeline.execute_predict(
                config=config,
                artifacts=artifacts,
                source_artifacts=source_artifacts,
                input_csv=input_csv,
            )
        else:
            generated = pipeline.execute_run(
                config=config,
                artifacts=artifacts,
                source_artifacts=source_artifacts,
                input_csv=input_csv,
            )

    write_research_manifest(
        artifacts=artifacts,
        command=command,
        config_path=config.config_path,
        generated_files={key: Path(value) for key, value in generated.items()},
    )
    write_latest_research_run(config.results_root, config.method_name, artifacts.run_id)
    print(f"[My_Capstone Research] completed run_id={artifacts.run_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_command(
        args.command,
        args.config,
        load_run=getattr(args, "load_run", None),
        input_csv=getattr(args, "input_csv", None),
    )


if __name__ == "__main__":
    raise SystemExit(main())
