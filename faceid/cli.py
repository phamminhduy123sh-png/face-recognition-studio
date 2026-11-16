"""Một CLI cho toàn bộ vòng đời: dữ liệu → train → hiệu chỉnh → test → nhận diện."""
import argparse
import json
from pathlib import Path
import sys

from .config import load_config


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _open_browser_when_ready(server, url):
    """Open a local browser only after this Uvicorn instance starts listening."""
    import time
    import webbrowser
    for _ in range(600):
        if server.should_exit:
            return
        if server.started:
            webbrowser.open(url)
            return
        time.sleep(0.05)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m faceid", description=__doc__)
    parser.add_argument("--config", type=Path, help="Cấu hình JSON, mặc định config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Tạo 18 thư mục và chuyển đổi trọng số cũ nếu có")
    sub.add_parser("audit", help="Kiểm tra dữ liệu, phiên và ảnh trùng trước train")
    sub.add_parser("import-legacy", help="Đóng gói trọng số cũ, chưa hiệu chỉnh/nghiệm thu")
    train_parser = sub.add_parser("train", help="Huấn luyện; chọn checkpoint bằng validation")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--backbone", choices=["facecnn", "resnet18"])
    train_parser.add_argument("--offline", action="store_true", help="Không tải trọng số pretrained")
    train_parser.add_argument("--warm-start", type=Path, help="Khởi tạo FaceCNN từ backbone cũ")
    calibration = sub.add_parser("calibrate", help="Chọn ngưỡng Unknown từ validation và người lạ")
    calibration.add_argument("--unknown-dir", type=Path, required=True)
    calibration.add_argument("--max-far", type=float, default=0.05)
    calibration.add_argument("--checkpoint", type=Path)
    evaluation = sub.add_parser("evaluate", help="Đánh giá checkpoint đã chọn; không train lại")
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--checkpoint", type=Path)
    predict = sub.add_parser("predict", help="Nhận diện các khuôn mặt trong một ảnh")
    predict.add_argument("image", type=Path)
    predict.add_argument("--checkpoint", type=Path)
    predict.add_argument("--threshold-file", type=Path)
    predict.add_argument("--manual-threshold", type=float, help="Ngưỡng thử tay, không được coi đã hiệu chỉnh")
    serve = sub.add_parser("serve", help="Mở giao diện web local")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--open-browser", action="store_true",
                       help="Tự mở trình duyệt sau khi server khởi động thành công")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "init":
            from .data import initialize_dataset
            folders = initialize_dataset(config["data_dir"])
            for name in ("unknown_val", "unknown_test"):
                (Path(config["data_dir"]).parent / name).mkdir(parents=True, exist_ok=True)
            Path(config["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
            result = {"session_folders": len(folders), "data_dir": config["data_dir"]}
            if (Path(config["legacy_dir"]) / "arcface_backbone.pth").is_file():
                from .engine import import_legacy
                result["legacy"] = import_legacy(config)
            _print(result)
        elif args.command == "audit":
            from .data import audit_dataset
            report = audit_dataset(config)
            _print(report)
            return 0 if report.get("valid") else 2
        elif args.command == "import-legacy":
            from .engine import import_legacy
            _print(import_legacy(config))
        elif args.command == "train":
            from .engine import train
            settings = config["training"]
            for key in ("epochs", "batch_size", "backbone"):
                if getattr(args, key) is not None:
                    settings[key] = getattr(args, key)
            if args.offline or settings["backbone"] == "facecnn":
                settings["pretrained"] = False
            if args.warm_start:
                settings["warm_start"] = str(args.warm_start.resolve())
            _print(train(config, progress=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True)))
        elif args.command == "calibrate":
            from .calibration import calibrate
            _print(calibrate(config, args.unknown_dir, max_far=args.max_far, checkpoint=args.checkpoint))
        elif args.command == "evaluate":
            from .engine import evaluate
            _print(evaluate(config, split=args.split, checkpoint=args.checkpoint))
        elif args.command == "predict":
            from PIL import Image
            from .inference import Predictor
            output = Path(config["artifacts_dir"])
            checkpoint = args.checkpoint or config.get("checkpoint") or output / "best.pth"
            if not Path(checkpoint).is_file() and not args.checkpoint:
                checkpoint = output / "legacy.pth"
            threshold = args.threshold_file or output / "threshold.json"
            predictor = Predictor(checkpoint, threshold_file=threshold if threshold.is_file() else None,
                                  device=config["training"]["device"], manual_threshold=args.manual_threshold)
            with Image.open(args.image) as image:
                _print(predictor.predict(image))
        elif args.command == "serve":
            import uvicorn
            from .web.app import create_app
            browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
            if ":" in browser_host:
                browser_host = f"[{browser_host}]"
            url = f"http://{browser_host}:{args.port}"
            print(f"Face Studio: {url}", flush=True)
            server = uvicorn.Server(uvicorn.Config(
                create_app(config["config_path"]), host=args.host, port=args.port))
            if args.open_browser:
                import threading
                threading.Thread(target=_open_browser_when_ready,
                                 args=(server, url), daemon=True).start()
            server.run()
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, OSError, ImportError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
