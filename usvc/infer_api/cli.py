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
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
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
    import uvicorn
    uvicorn.run(create_app(root, device=device, token=args.token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
