"""Orchestrator: build year corpora, generate synthetic SFT, mid+SFT train, probe.

Stages run independently so a failure in one year does not block the others until
training, which is necessarily sequential because year Y starts from year Y-1's ckpt.

    python -m policy_pred.pipeline slice --start 1931 --end 1940
    python -m policy_pred.pipeline synth --start 1931 --end 1940
    python -m policy_pred.pipeline train --start 1931 --end 1940
    python -m policy_pred.pipeline eval  --start 1931 --end 1940

To populate a base model into its registered weights_dir on D::
    python -m policy_pred.pipeline setup-model talkie_base
    python -m policy_pred.pipeline setup-model nanochat_1925 --from-path <ckpt>
"""
import argparse
from pathlib import Path

from . import config
from .corpus import year_slice
from .synthesize import generate
from .train import midtrain, sft
from .eval import elicit, analyze


def cmd_slice(args):
    for year in range(args.start, args.end + 1):
        year_slice.build(year)


def cmd_synth(args):
    for year in range(args.start, args.end + 1):
        generate.generate_year(year)


def cmd_train(args):
    """Cumulative training: each year picks up from the prior year's SFT checkpoint.

    The model spec is carried forward year by year, with only the 'path' field
    replaced to point at each new checkpoint. Backend (nanochat vs Talkie) is
    determined once by the BASE_MODEL setting in config.py.
    """
    spec = config.base_model_spec()
    for year in range(args.start, args.end + 1):
        mid_ckpt = midtrain.run(year, init_spec=spec)
        sft_ckpt = sft.run(year, init_spec={**spec, "path": str(mid_ckpt)})
        spec = {**spec, "path": str(sft_ckpt)}


def cmd_eval(args):
    for year in range(args.start, args.end + 1):
        elicit.run_all_policies(year)
    analyze.build_trajectories()


def cmd_setup_model(args):
    """Populate a registry entry's weights_dir on D:.

    'talkie' entries: snapshot_download from HF into weights_dir.
    'nanochat' entries: copy or symlink from --from-path.
    """
    from .models.registry import load_registry

    entry = load_registry()[args.name]
    target = Path(entry["weights_dir"])
    target.mkdir(parents=True, exist_ok=True)
    kind = entry["type"]
    if kind == "talkie":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=entry["hf_id"], local_dir=str(target))
        print(f"downloaded {entry['hf_id']} -> {target}")
    elif kind == "nanochat":
        if not args.from_path:
            raise SystemExit("nanochat models need --from-path <source ckpt dir>")
        src = Path(args.from_path).resolve()
        if any(target.iterdir()):
            raise SystemExit(f"{target} already populated; remove it first")
        if args.copy:
            import shutil
            shutil.copytree(src, target, dirs_exist_ok=True)
            print(f"copied {src} -> {target}")
        else:
            target.rmdir()
            target.symlink_to(src, target_is_directory=True)
            print(f"linked {src} -> {target}")
    else:
        raise SystemExit(f"don't know how to populate type={kind!r}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in [
        ("slice", cmd_slice),
        ("synth", cmd_synth),
        ("train", cmd_train),
        ("eval", cmd_eval),
    ]:
        sp = sub.add_parser(name)
        sp.add_argument("--start", type=int, default=config.START_YEAR)
        sp.add_argument("--end", type=int, default=config.END_YEAR)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("setup-model")
    sp.add_argument("name", help="registry name (e.g. talkie_base, nanochat_1925)")
    sp.add_argument("--from-path", help="(nanochat) source checkpoint directory")
    sp.add_argument("--copy", action="store_true", help="(nanochat) copy instead of symlink")
    sp.set_defaults(func=cmd_setup_model)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
