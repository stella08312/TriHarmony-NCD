"""Train and decode the fixed Full method without evaluation labels."""
from pathlib import Path
from .config import load_config
from .utils import (canonical_hash, initialize, new_output_directory,
                    prepare_environment, runtime_record, sha256_file,
                    torch_load, write_json)

def _setup(args):
    config = load_config(args.config, args.seed, args.cpu_threads)
    prepare_environment(config)
    device = initialize(config, args.device)
    return config, device

def _evidence_numpy(target):
    return {name: value.float().cpu().numpy()
            for name, value in target["evidence"].items()}

def _prediction(target, learned, config, checkpoint_sha):
    result = {
        "schema_version": 1, "method": config["method"], "variant": "learned",
        "protocol": target["protocol"], "dataset": target["dataset"],
        "category": target["category"], "sample_keys": target["keys"],
        "seed": config["seed"], "sole_numeric_seed": config["seed"],
        "epoch": 40, "checkpoint": "fixed_epoch_40",
        "checkpoint_sha256": checkpoint_sha, "target_gt_used": False,
        "target_metric_used": False, "target_normal_used": False,
        "known_k": target["known_k"], "additional_numeric_seeds": [],
        "same_configuration_all_jobs": True, "all_graph_channels_always_computed": True,
        "method_graph_channels": 4, **learned,
    }
    result["numeric_seeds_used_by_decoder"] = [config["seed"]]
    from .evaluation import validate_prediction
    validate_prediction(result)
    return result

def _finish(output, target, learned, config, checkpoint_path, manifest):
    prediction = _prediction(target, learned, config, sha256_file(checkpoint_path))
    write_json(output / "learned_predictions.json", prediction)
    digest = sha256_file(output / "learned_predictions.json")
    write_json(output / "predictions_manifest.json", {
        "schema_version": 1, "runs": [
            {"predictions": "learned_predictions.json", "sha256": digest}
        ],
    })
    write_json(output / "run_manifest.json", {
        "status": "complete", "fixed_final_checkpoint": True,
        "prediction_sha256": digest, "checkpoint_sha256": sha256_file(checkpoint_path),
        "protocol": target["protocol"], "dataset": target["dataset"],
        "category": target["category"], "samples": len(target["keys"]),
        "known_k": target["known_k"], "seed": config["seed"],
        "target_gt_used_for_training_inference_or_selection": False,
        "target_metric_used_for_training_inference_or_selection": False,
        **manifest,
    })
    return {"output": str(output), "predictions": str(output / "learned_predictions.json")}

def train(args):
    config, device = _setup(args)
    import torch
    from .inputs import load_source, load_target
    from .training import prepare_source, train_one
    source, target = load_source(args.source), load_target(args.target)
    output = new_output_directory(args.output)
    write_json(output / "config.json", config)
    prepared = prepare_source(source, device)
    model, training, learned = train_one(prepared, target, device)
    checkpoint_path = output / "checkpoint_epoch40.pt"
    torch.save({
        "schema_version": 1, "method": config["method"], "epoch": 40,
        "seed": config["seed"], "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "target_gt_used": False, "target_metric_used": False,
        "config_sha256": canonical_hash(config),
        "target_sha256": sha256_file(args.target),
        "source_sha256": sha256_file(args.source),
        "runtime": runtime_record(config, device),
    }, checkpoint_path)
    write_json(output / "training.json", training)
    return _finish(output, target, learned, config, checkpoint_path, {
        "operation": "train", "runtime": runtime_record(config, device),
        "config_sha256": canonical_hash(config),
        "source_sha256": sha256_file(args.source), "target_sha256": sha256_file(args.target),
        "training_sha256": sha256_file(output / "training.json"),
        "epochs": training["epochs"],
        "relation_module_trainable_parameters": training["relation_module_trainable_parameters"],
    })

def infer(args):
    config, device = _setup(args)
    from .inputs import load_target
    from .model import PersistentGraphTransport, target_graph_bundle, learned_partition
    target = load_target(args.target)
    checkpoint = torch_load(args.checkpoint)
    expected = {
        "schema_version": 1, "method": config["method"], "epoch": 40,
        "seed": config["seed"], "target_gt_used": False, "target_metric_used": False,
        "config_sha256": canonical_hash(config), "target_sha256": sha256_file(args.target),
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError("Checkpoint/input/configuration mismatch: %s" % name)
    model = PersistentGraphTransport().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    evidence = _evidence_numpy(target)
    bundle = target_graph_bundle(evidence, target["known_k"],
                                 target["anomaly_support"].cpu().numpy())
    prediction, decoder = learned_partition(model, bundle, evidence, target["known_k"], device)
    learned = {
        "prediction": prediction.tolist(), "occupied_k": int(len(set(prediction.tolist()))),
        "decoder": decoder, "additional_numeric_seeds_used": [],
        "target_gt_used": False, "target_metric_used": False,
    }
    output = new_output_directory(args.output)
    write_json(output / "config.json", config)
    return _finish(output, target, learned, config, Path(args.checkpoint), {
        "operation": "infer", "runtime": runtime_record(config, device),
        "checkpoint_runtime": checkpoint.get("runtime"),
        "config_sha256": canonical_hash(config), "target_sha256": sha256_file(args.target),
    })
