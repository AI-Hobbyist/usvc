"""Bounded availability worker; only CPU metadata leaves this process."""
import argparse
import json
from pathlib import Path
from .catalog import Catalog
from .runtime import Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--enhancer", action="store_true")
    args = parser.parse_args()
    runtime = Runtime(args.device)
    error = None
    try:
        catalog = Catalog(args.model_root, lambda model: None)
        if args.enhancer:
            backend, name = args.model.split("/")
            directory = (catalog.root / backend / name).resolve()
            if backend != "reflow_shallow" or not directory.is_relative_to(catalog.root):
                raise ValueError("invalid enhancer ID")
            model = catalog._read(directory, backend)
            runtime.probe_enhancer_local(model)
        else:
            model = catalog.candidate(args.model)
            runtime.probe_local(model)
    except Exception as exc:
        error = str(exc)
    finally:
        runtime.close()
        args.report.write_text(json.dumps(dict(history=runtime.history, error=error), ensure_ascii=False, indent=2), encoding="utf-8")
    if error:
        print(error, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
