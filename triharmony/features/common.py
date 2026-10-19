from __future__ import annotations
import hashlib, math, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision import transforms

@dataclass(frozen=True)
class Sample:
    image_path: Path
    map_path: Optional[Path]
    label: str
    category: str
    dataset: str
    protocol: str
    source_label: Optional[str] = None


@dataclass
class Batch:
    tokens224: torch.Tensor
    tokens448: torch.Tensor
    map224: torch.Tensor
    map448: torch.Tensor
    coords224: torch.Tensor
    coords448: torch.Tensor
    eval_label: List[str]
    source_label: Optional[torch.Tensor]
    normal_tokens: Optional[torch.Tensor]
    meta: List[Dict[str, object]]

    @property
    def device(self) -> torch.device:
        return self.tokens224.device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), copy=False)
    arr -= float(arr.min())
    mx = float(arr.max())
    if mx > 1e-6:
        arr /= mx
    return arr.astype(np.float32)


def read_map(path: Optional[Path], size: int) -> torch.Tensor:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing anomaly map: {path}")
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=0) if arr.shape[0] <= 4 else arr.mean(axis=-1)
    else:
        image = Image.open(path)
        # Preserve high-bit-depth score maps (notably the official PNI MVTec
        # 16-bit PNGs). PIL's ``I;16 -> L`` conversion clips every value above
        # 255, which can collapse a valid anomaly map to a constant image.
        if image.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
            arr = np.asarray(image).astype(np.float32)
        else:
            arr = np.asarray(image.convert("L")).astype(np.float32)
    ten = torch.from_numpy(normalize_map(arr))[None, None]
    ten = F.interpolate(ten, size=(size, size), mode="bilinear", align_corners=False)[0]
    ten = ten.clamp(0, 1)
    if float(ten.std()) < 1e-8:
        ten = ten + 1e-4 * torch.rand_like(ten)
    return ten


def read_mask_as_map(path: Path, size: int) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L")).astype(np.float32)
    arr = (arr > 15.0).astype(np.float32)
    ten = torch.from_numpy(arr)[None, None]
    ten = F.interpolate(ten, size=(size, size), mode="bilinear", align_corners=False)[0]
    return ten.clamp(0, 1)


class DINOv3PatchExtractor:
    MODEL_SPECS = {
        "dinov3_vitb16": ("vit_base_patch16_dinov3", 768),
    }

    def __init__(self, device: str, image_size: int, token_dim: int, seed: int, batch_size: int, backend: str):
        if backend not in self.MODEL_SPECS:
            raise ValueError(f"Unsupported DINOv3 backend: {backend}")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.token_dim = int(token_dim)
        self.batch_size = int(batch_size)
        self.backend = backend
        self.model_name, self.source_dim = self.MODEL_SPECS[backend]
        self.model = self._load_model().to(self.device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.proj = None
        if self.token_dim != self.source_dim:
            stable = int(hashlib.sha1(f"{backend}:{self.source_dim}:{self.token_dim}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed + stable % 1000003)
            proj = rng.normal(0.0, 1.0 / math.sqrt(max(1, self.token_dim)), size=(self.source_dim, self.token_dim)).astype(np.float32)
            self.proj = torch.from_numpy(proj).to(self.device)

    def _load_model(self):
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("DINOv3 backend requires timm in the active Python environment.") from exc
        checkpoint = getattr(type(self), "checkpoint_path", None)
        if checkpoint is None:
            raise ValueError("A local DINOv3 safetensors checkpoint is required.")
        with torch.random.fork_rng(devices=[]):
            return timm.create_model(self.model_name, pretrained=True,
                pretrained_cfg_overlay={"file": str(checkpoint)})

    @torch.no_grad()
    def pil_batch_tokens(self, images: Sequence[Image.Image]) -> Tuple[np.ndarray, int]:
        tensors = [self.preprocess(img.convert("RGB")) for img in images]
        rows: List[np.ndarray] = []
        side = 1
        for start in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[start : start + self.batch_size], 0).to(self.device)
            out = self.model.forward_features(batch)
            if isinstance(out, dict):
                patch = out.get("x_norm_patchtokens")
                if patch is None:
                    patch = out.get("x_prenorm")
            else:
                patch = out
            if patch is None:
                raise RuntimeError(f"Could not extract patch tokens from {self.model_name}.")
            if patch.ndim != 3:
                raise RuntimeError(f"Expected token tensor [B, N, D] from {self.model_name}, got shape {tuple(patch.shape)}.")
            side_from_all = int(math.sqrt(patch.shape[1]))
            if side_from_all * side_from_all != patch.shape[1]:
                n_prefix = int(getattr(self.model, "num_prefix_tokens", 1))
                patch = patch[:, n_prefix:, :]
            side = int(math.sqrt(patch.shape[1]))
            if side * side != patch.shape[1]:
                raise RuntimeError(f"Patch token count is not square for {self.model_name}: {patch.shape[1]}")
            if self.proj is not None:
                patch = patch @ self.proj
            rows.append(F.normalize(patch, dim=-1).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(rows, 0), side


class FeatureCache:
    def __init__(
        self,
        cache_dir: Path,
        device: torch.device,
        image_size: int,
        hr_image_size: int,
        token_dim: int,
        backend: str,
        batch_size: int,
        seed: int,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.image_size = int(image_size)
        self.hr_image_size = int(hr_image_size)
        self.token_dim = int(token_dim)
        self.backend = backend
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._extractors: Dict[int, Optional[object]] = {}

    def _extractor(self, image_size: int) -> object:
        if image_size in self._extractors:
            return self._extractors[image_size]
        if self.backend != "dinov3_vitb16":
            raise ValueError(
                f"The Main-Table supplement supports only dinov3_vitb16, got {self.backend!r}."
            )
        self._extractors[image_size] = DINOv3PatchExtractor(
            str(self.device), image_size, self.token_dim, self.seed, self.batch_size, self.backend
        )
        return self._extractors[image_size]

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.npz"

    def tokens_for_image(self, image_path: Path, image_size: int, target_side: int) -> torch.Tensor:
        cache = self._cache_path(f"{image_path}:{image_size}:{target_side}:{self.token_dim}:{self.backend}:v2")
        if cache.exists():
            try:
                return torch.from_numpy(np.load(cache)["tokens"].astype(np.float32)).to(self.device)
            except Exception:
                cache.unlink(missing_ok=True)
        extractor = self._extractor(image_size)
        toks, side = extractor.pil_batch_tokens([Image.open(image_path).convert("RGB")])
        token = torch.from_numpy(toks[0]).float()
        if side != target_side:
            token = token.reshape(side, side, -1).permute(2, 0, 1)[None]
            token = F.interpolate(token, size=(target_side, target_side), mode="bilinear", align_corners=False)
            token = token[0].permute(1, 2, 0).reshape(target_side * target_side, -1)
        token = F.normalize(token, dim=-1)
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, tokens=token.detach().cpu().numpy().astype(np.float32))
        os.replace(tmp, cache)
        return token.to(self.device)

    def coords(self, side: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(torch.linspace(0, 1, side), torch.linspace(0, 1, side), indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(self.device)

    def _bbox_from_map(self, amap: torch.Tensor, q: float = 0.85, pad_ratio: float = 0.18) -> Tuple[float, float, float, float]:
        arr = amap[0].detach().cpu().numpy()
        h, w = arr.shape
        thr = max(float(np.quantile(arr, q)), float(arr.mean() + 0.35 * arr.std()))
        mask = arr >= thr
        labels, n_comp = ndimage.label(mask)
        if n_comp <= 0:
            y, x = np.unravel_index(int(arr.argmax()), arr.shape)
            half = max(8, min(h, w) // 8)
            y0, y1 = max(0, y - half), min(h, y + half)
            x0, x1 = max(0, x - half), min(w, x + half)
        else:
            best_lab, best_score = 1, -1.0
            for lab in range(1, n_comp + 1):
                comp = labels == lab
                score = float(arr[comp].sum())
                if score > best_score:
                    best_lab, best_score = lab, score
            ys, xs = np.where(labels == best_lab)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
        pad = int(round(max(x1 - x0, y1 - y0, h, w) * pad_ratio))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        return x0 / w, y0 / h, x1 / w, y1 / h

    def roi_crop(self, image_path: Path, amap: torch.Tensor, tag: str) -> Tuple[Path, torch.Tensor, List[float]]:
        box = self._bbox_from_map(amap)
        digest = hashlib.sha1(f"{image_path}:{tag}:{box}:{self.hr_image_size}".encode()).hexdigest()[:24]
        crop_dir = self.cache_dir / "roi_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"{digest}.png"
        if not crop_path.exists():
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            x0, y0, x1, y1 = box
            crop = img.crop((int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 1), max(int(y1 * h), int(y0 * h) + 1)))
            crop.resize((self.hr_image_size, self.hr_image_size), Image.BICUBIC).save(crop_path)
        mh, mw = amap.shape[-2:]
        x0, y0, x1, y1 = box
        mx0, my0, mx1, my1 = int(x0 * mw), int(y0 * mh), max(int(x1 * mw), int(x0 * mw) + 1), max(int(y1 * mh), int(y0 * mh) + 1)
        crop_map = amap[:, my0:my1, mx0:mx1][None]
        crop_map = F.interpolate(crop_map, size=(self.hr_image_size, self.hr_image_size), mode="bilinear", align_corners=False)[0].clamp(0, 1)
        return crop_path, crop_map, [float(v) for v in box]

    def normal_tokens(self, image_paths: Sequence[Path], side: int) -> Optional[torch.Tensor]:
        if not image_paths:
            return None
        rows = []
        for p in image_paths:
            tok = self.tokens_for_image(p, self.image_size, side)
            rows.append(tok.mean(dim=0))
        return F.normalize(torch.stack(rows, 0), dim=-1)

    def build_batch(self, samples: Sequence[Sample], args, normal_images: Optional[Sequence[Path]] = None) -> Batch:
        t224, t448, m224, m448, labels, meta = [], [], [], [], [], []
        source_labels: List[int] = []
        source_names = sorted({s.source_label for s in samples if s.source_label is not None})
        source_map = {name: idx for idx, name in enumerate(source_names)}
        for s in samples:
            if s.protocol == "source":
                amap = read_mask_as_map(s.map_path, args.image_size)
                feature_path = s.image_path
                roi_path = s.image_path
                roi_map = read_mask_as_map(s.map_path, args.hr_image_size)
                bbox = [0.0, 0.0, 1.0, 1.0]
            else:
                amap = read_map(s.map_path, args.image_size)
                roi_path, roi_map, bbox = self.roi_crop(s.image_path, amap, f"{s.protocol}_{s.dataset}_{s.category}_{s.label}")
                feature_path = s.image_path
            tok224 = self.tokens_for_image(feature_path, args.image_size, args.grid224)
            tok448 = self.tokens_for_image(roi_path, args.hr_image_size, args.grid448)
            t224.append(tok224)
            t448.append(tok448)
            m224.append(F.interpolate(amap[None], size=(args.grid224, args.grid224), mode="area")[0, 0].flatten())
            m448.append(F.interpolate(roi_map[None], size=(args.grid448, args.grid448), mode="area")[0, 0].flatten())
            labels.append(s.label)
            if s.source_label is not None:
                source_labels.append(source_map[s.source_label])
            meta.append(
                {
                    "image": str(s.image_path),
                    "map": str(s.map_path) if s.map_path else "",
                    "roi_image": str(roi_path),
                    "roi_bbox_xyxy_norm": bbox,
                    "label_eval_only": s.label,
                    "category": s.category,
                    "dataset": s.dataset,
                    "protocol": s.protocol,
                }
            )
        src_tensor = None
        if source_labels:
            src_tensor = torch.tensor(source_labels, dtype=torch.long, device=self.device)
        normal = None
        if normal_images is not None:
            normal = self.normal_tokens(normal_images, args.grid224)
        return Batch(
            tokens224=F.normalize(torch.stack(t224, 0).to(self.device), dim=-1),
            tokens448=F.normalize(torch.stack(t448, 0).to(self.device), dim=-1),
            map224=torch.stack(m224, 0).to(self.device),
            map448=torch.stack(m448, 0).to(self.device),
            coords224=self.coords(args.grid224),
            coords448=self.coords(args.grid448),
            eval_label=labels,
            source_label=src_tensor,
            normal_tokens=normal,
            meta=meta,
        )
