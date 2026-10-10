"""List models, read settings and archive the exact binary inference response."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def workflow(base_url, source, output_dir, model=None, params=None, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def request(path, body=None):
        options = dict(headers)
        if body is not None:
            options["Content-Type"] = "audio/wav"
        try:
            with urlopen(Request(base_url.rstrip("/") + path, data=body, headers=options), timeout=600) as response:
                return response.read(), {key.lower(): value for key, value in response.headers.items()}
        except HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode('utf-8')}") from None

    raw, _ = request("/model")
    models = json.loads(raw)
    print(json.dumps(models, ensure_ascii=False, indent=2), flush=True)
    selected = next((item for item in models if item["id"] == model), None) if model else next(iter(models), None)
    if selected is None:
        raise ValueError("Requested model is unavailable")
    raw, _ = request("/settings?" + urlencode({"backend": selected["backend"], "model": selected["id"]}))
    settings = json.loads(raw)
    print(json.dumps(settings, ensure_ascii=False, indent=2), flush=True)
    values = dict(params or {})
    if selected["multi_speaker"] and "speaker" not in values:
        values["speaker"] = selected["speakers"][0]["id"]
    source_bytes = Path(source).read_bytes()
    query = urlencode({"model": selected["id"], "params": json.dumps(values, ensure_ascii=False)})
    wav, response_headers = request("/infer?" + query, source_bytes)
    if response_headers.get("content-type", "").split(";")[0] != "audio/wav":
        raise ValueError("Expected an audio/wav response")
    digest = hashlib.sha256(wav).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{digest}.wav"
    if not path.exists():
        path.write_bytes(wav)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Existing archive does not match its filename")
    result = dict(model=selected["id"], params=values, device=settings["device"],
                  source_sha256=hashlib.sha256(source_bytes).hexdigest(), output_sha256=digest,
                  path=str(path.resolve()), headers=response_headers)
    path.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", default="tests/input/1.wav")
    parser.add_argument("--output-dir", default="tests/output")
    parser.add_argument("--model")
    parser.add_argument("--params", default="{}")
    parser.add_argument("--params-file", type=Path, help="UTF-8 JSON file; avoids native shell quote conversion")
    parser.add_argument("--token")
    args = parser.parse_args()
    values = json.loads(args.params_file.read_text(encoding="utf-8-sig") if args.params_file else args.params)
    workflow(args.base_url, args.input, args.output_dir, args.model, values, args.token)
