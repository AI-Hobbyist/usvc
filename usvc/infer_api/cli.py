import argparse
import ipaddress
from pathlib import Path
import socket


def main(argv=None):
    parser = argparse.ArgumentParser(description="USVC independent offline inference API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--token", default=None)
    parser.add_argument("--max-upload-bytes", type=int, default=64 * 1024 ** 2)
    parser.add_argument("--max-audio-seconds", type=float, default=600.)
    parser.add_argument("--max-pending", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if min(args.max_upload_bytes, args.max_audio_seconds, args.max_pending) <= 0:
        parser.error("upload, duration and scheduler limits must be positive")
    try:
        ipaddress.ip_address(args.host)
        root = args.model_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("model root must be a directory")
        import torch
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
        torch.empty(0, device=device)
        family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
        with socket.socket(family) as listener:
            listener.bind((args.host, args.port))
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        parser.error(str(exc))
    from .app import create_app
    import logging
    logging.basicConfig(level=logging.INFO)
    import uvicorn
    uvicorn.run(create_app(root, device=device, token=args.token, max_upload_bytes=args.max_upload_bytes,
                           max_audio_seconds=args.max_audio_seconds, max_pending=args.max_pending), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
