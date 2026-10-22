from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from logging import getLogger
from os import cpu_count
from pathlib import Path
from typing import Union

import cricodecs
from ..image_sizes import image_for_export, sprite_for_export, image_output_paths, texture_canvas_sizes, texture_image
from PIL import Image

from .. import crypto
from . import asset_bucket, kind_rel

logger = getLogger("extract")

UNITY_VERSION = "6000.3.15f1"

_KEEP = (
    "Sprite",
    "Texture2D",
    "TextAsset",
    "MonoBehaviour",
    "AnimationClip",
    "Mesh",
    "SkinnedMeshRenderer",
    "MeshRenderer",
    "MeshFilter",
    "LODGroup",
    "Transform",
    "GameObject",
    "Material",
    "Shader",
    "Avatar",
    "AnimatorController",
    "AnimatorOverrideController",
)


def _asset_name(data, path_id: int) -> str:
    return getattr(data, "m_Name", None) or str(path_id)


def _claim_stem(used: set[str], name: str, path_id: int, suffixes: tuple[str, ...]) -> str:
    base = name or str(path_id)
    n = 0
    while True:
        stem = base if n == 0 else f"{base}__{path_id}" if n == 1 else f"{base}__{path_id}_{n - 1}"
        filenames = [f"{stem}{suffix}" for suffix in suffixes]
        if all(filename not in used for filename in filenames):
            used.update(filenames)
            return stem
        n += 1


def _claim(out: Path, used: set[str], name: str, path_id: int, suffix: str) -> Path:
    dest = out / f"{_claim_stem(used, name, path_id, (suffix,))}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _num(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _vec3(value) -> list | None:
    if value is None:
        return None
    return [_num(value.x), _num(value.y), _num(value.z)]


def _matrix(value) -> list:
    return [_num(getattr(value, f"e{row}{col}")) for row in range(4) for col in range(4)]


def _rows(rows) -> list:
    if not rows:
        return []
    return [[_num(item) for item in row] for row in rows]


def _blend_shapes(shapes) -> dict:
    if not shapes:
        return {"channels": [], "shapes": [], "fullWeights": [], "vertices": []}
    return {
        "channels": [
            {
                "name": channel.name,
                "nameHash": _num(channel.nameHash),
                "frameIndex": _num(channel.frameIndex),
                "frameCount": _num(channel.frameCount),
            }
            for channel in (shapes.channels or [])
        ],
        "shapes": [
            {
                "firstVertex": _num(shape.firstVertex),
                "vertexCount": _num(shape.vertexCount),
                "hasNormals": bool(shape.hasNormals),
                "hasTangents": bool(shape.hasTangents),
            }
            for shape in (shapes.shapes or [])
        ],
        "fullWeights": [_num(weight) for weight in (shapes.fullWeights or [])],
        "vertices": [
            {
                "index": _num(vertex.index),
                "vertex": _vec3(vertex.vertex),
                "normal": _vec3(vertex.normal),
                "tangent": _vec3(vertex.tangent),
            }
            for vertex in (shapes.vertices or [])
        ],
    }


def _mesh_skin(mesh) -> dict:
    from UnityPy.helpers.MeshHelper import MeshHandler

    handler = MeshHandler(mesh)
    handler.process()
    return {
        "bindPoses": [_matrix(pose) for pose in (mesh.m_BindPose or [])],
        "boneNameHashes": [_num(item) for item in (mesh.m_BoneNameHashes or [])],
        "boneWeights": _rows(handler.m_BoneWeights),
        "boneIndices": _rows(handler.m_BoneIndices),
        "blendShapes": _blend_shapes(mesh.m_Shapes),
    }


def _shader_text(shader) -> str | None:
    try:
        text = shader.export()
    except Exception:
        return None
    if not isinstance(text, str) or not text:
        return None
    return text


def _write_json(dest: Path, data) -> None:
    dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _unitypy():
    import warnings

    import UnityPy
    from UnityPy.exceptions import UnityVersionFallbackWarning

    UnityPy.config.FALLBACK_UNITY_VERSION = UNITY_VERSION
    warnings.simplefilter("ignore", UnityVersionFallbackWarning)
    return UnityPy


class AssetBundle:
    def __init__(self, path: Union[str, Path], name: str) -> None:
        self.path = Path(path)
        self.name = name

    def load(self):
        return _unitypy().load(crypto.decrypt(self.path.read_bytes(), self.name))

    def extract(self, outdir: Union[str, Path], images_only: bool = False) -> set[Path]:
        env = self.load()
        out = Path(outdir) / self.name
        keep = ("Sprite", "Texture2D") if images_only else _KEEP
        objects = [(o, o.read()) for o in env.objects if o.type.name in keep]
        sprites = {(id(o.assets_file), d.m_Name) for o, d in objects if o.type.name == "Sprite"}
        objects = [(o, d) for o, d in objects if not (
            o.type.name == "Texture2D" and (id(o.assets_file), d.m_Name) in sprites)]
        # Runtime-created font atlases have no pixels and cannot be saved as PNG.
        objects = [(o, d) for o, d in objects if not (
            o.type.name == "Texture2D" and (d.m_Width <= 0 or d.m_Height <= 0))]
        image_paths = image_output_paths([
            dict(serialized_file=o.assets_file.name, id=o.path_id, name=d.m_Name,
                 size=(d.m_Width, d.m_Height) if o.type.name == "Texture2D" else (d.m_Rect.width, d.m_Rect.height))
            for o, d in objects if o.type.name in ("Sprite", "Texture2D")])
        canvases = texture_canvas_sizes(env) if image_paths else {}
        created: set[Path] = set()
        used = {path.name for path in image_paths.values()}
        for obj, d in objects:
            name = _asset_name(d, obj.path_id)
            if obj.type.name in ("Sprite", "Texture2D"):
                dest = out / image_paths[(obj.assets_file.name, obj.path_id)]
                dest.parent.mkdir(parents=True, exist_ok=True)
                image = sprite_for_export(d) if obj.type.name == "Sprite" else image_for_export("Texture2D", d.m_Name, texture_image(d))
                canvas = canvases.get((obj.assets_file.name, obj.path_id))
                if obj.type.name == "Texture2D" and canvas and image.size != canvas:
                    image = d.image.resize(canvas, Image.Resampling.LANCZOS)
                image.save(dest)
                created.add(dest)
            elif obj.type.name == "TextAsset":
                dest = _claim(out, used, name, obj.path_id, "")
                script = d.m_Script
                dest.write_bytes(
                    script.encode("utf-8", "surrogateescape")
                    if isinstance(script, str)
                    else bytes(script)
                )
                created.add(dest)
            elif obj.type.name == "Mesh":
                stem = _claim_stem(used, name, obj.path_id, (".obj", ".skin.json"))
                obj_path = out / f"{stem}.obj"
                skin_path = out / f"{stem}.skin.json"
                obj_path.parent.mkdir(parents=True, exist_ok=True)
                text = d.export()
                if isinstance(text, str) and text:
                    obj_path.write_text(text, encoding="utf-8", newline="")
                    created.add(obj_path)
                _write_json(skin_path, _mesh_skin(d))
                created.add(skin_path)
            elif obj.type.name == "Shader":
                text = _shader_text(d)
                if text is None:
                    dest = _claim(out, used, name, obj.path_id, ".json")
                    _write_json(dest, obj.read_typetree())
                else:
                    dest = _claim(out, used, name, obj.path_id, ".shader")
                    dest.write_text(text, encoding="utf-8", newline="")
                created.add(dest)
            else:
                dest = _claim(out, used, name, obj.path_id, ".json")
                _write_json(dest, obj.read_typetree())
                created.add(dest)
        return created


def _wav(name: str, data: bytes) -> bytes:
    if name.endswith(".adx"):
        adx = cricodecs.adx.load_bytes(data)
        pcm = adx.decode()
        if pcm[:4] == b"RIFF":
            return pcm
        h = adx.header
        return cricodecs.wav.build_bytes(pcm, h.sample_rate, h.channels)
    return cricodecs.hca.decode(data)


def _extract_acb(src: Path, outdir: Path) -> set[Path]:
    acb = cricodecs.acb.load(str(src))
    count = acb.waveform_count
    created: set[Path] = set()
    for i in range(count):
        wav = _wav(acb.waveform_filename(i), acb.waveform_bytes(i))
        if count == 1:
            dest = outdir / (src.stem + ".wav")
        else:
            dest = outdir / src.stem / (Path(acb.waveform_filename(i)).stem + ".wav")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(wav)
        created.add(dest)
    return created


def _extract_usm(src: Path, outdir: Path) -> set[Path]:
    usm = cricodecs.usm.load(str(src))
    out = outdir / src.stem
    out.mkdir(parents=True, exist_ok=True)
    usm.extract(str(out))
    created: set[Path] = set()
    for f in sorted(out.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() in (".hca", ".adx"):
            wav = f.with_suffix(".wav")
            wav.write_bytes(_wav(f.name, f.read_bytes()))
            f.unlink()
            created.add(wav)
        else:
            created.add(f)
    return created


def _copy(src: Path, outdir: Path) -> set[Path]:
    dest = outdir / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return {dest}


def _extract_one(src: Path, outdir: Path, images_only: bool = False) -> set[Path]:
    ext = src.suffix.lower()
    if images_only:
        return AssetBundle(src, src.name).extract(outdir, images_only=True)
    if ext == ".acb":
        return _extract_acb(src, outdir)
    if ext == ".usm":
        return _extract_usm(src, outdir)
    if ext == ".awb":
        return set() if src.with_suffix(".acb").exists() else _copy(src, outdir)
    with open(src, "rb") as fp:
        head = fp.read(7)
    if ext == "" or head == b"UnityFS":
        return AssetBundle(src, src.name).extract(outdir)
    return _copy(src, outdir)


def _extract_job(src: Path, outdir: Path, images_only: bool = False) -> tuple[str, int, str | None]:
    try:
        return src.name, len(_extract_one(src, outdir, images_only)), None
    except Exception as e:
        return src.name, 0, str(e)


def _log_extract(name: str, count: int, err: str | None) -> None:
    if err:
        logger.error("%s failed: %s", name, err)
    else:
        logger.info("%s -> %d files", name, count)


def _consume(
    files: list[Path], outdir: Path, workers: int, images_only: bool = False
) -> tuple[set[Path], bool, list[tuple[Path, int, str | None]]]:
    done: set[Path] = set()
    results: list[tuple[Path, int, str | None]] = []
    if not files:
        return done, False, results
    n = workers if workers > 0 else (cpu_count() or 4)
    pool = ProcessPoolExecutor(max_workers=min(n, len(files)))
    broken = False
    futs = {
        pool.submit(_extract_job, src, outdir / kind_rel(src), images_only): src for src in files
    }

    def _take(src: Path, name: str, count: int, err: str | None) -> None:
        done.add(src)
        results.append((src, count, err))
        _log_extract(name, count, err)

    try:
        try:
            for fut in as_completed(futs):
                src = futs[fut]
                try:
                    name, count, err = fut.result()
                except BrokenProcessPool:
                    broken = True
                    break
                _take(src, name, count, err)
        except BrokenProcessPool:
            broken = True
        for fut, src in futs.items():
            if src in done or not fut.done():
                continue
            try:
                name, count, err = fut.result(timeout=0)
            except Exception:
                continue
            _take(src, name, count, err)
    finally:
        try:
            pool.shutdown(wait=not broken, cancel_futures=broken)
        except Exception:
            pass
    return done, broken, results


def extract_many(
    files: list[Path], outdir: Path, workers: int = 0, *, images_only: bool = False
) -> list[tuple[Path, int, str | None]]:
    pending = list(files)
    if not pending:
        return []
    done, broken, results = _consume(pending, outdir, workers, images_only)
    if not broken:
        return results
    # ponytail: after one killed worker, the rest run one at a time
    pending = [src for src in pending if src not in done]
    logger.error("worker killed; retrying %d file(s) one at a time", len(pending))
    for src in pending:
        _, one_broken, one_results = _consume([src], outdir, 1, images_only)
        results.extend(one_results)
        if one_broken:
            logger.error("%s killed the worker; skipped", src.name)
    return results


def main(args: argparse.Namespace) -> int:
    indir, outdir = Path(args.indir), Path(args.outdir)
    files = (
        [indir]
        if indir.is_file()
        else sorted(p for p in indir.rglob("*") if p.is_file())
    )
    extract_many(files, outdir, args.workers)
    return 0


if __name__ == "__main__":
    name, n, err = _extract_job(Path("_missing_no_such_file"), Path("."))
    assert err and n == 0 and name == "_missing_no_such_file"
    assert extract_many([], Path(".")) == []
    assert asset_bucket("adv_anime_01_01_01-01") == "adv"
    assert asset_bucket("VisionProject.acf") == "_"
    assert kind_rel(Path("assets/assetbundles/adv/adv_anime_01_01_01-01")) == Path(
        "assetbundles", "adv"
    )
    assert kind_rel(Path("assets/assetbundles/adv_anime_01_01_01-01")) == Path(
        "assetbundles", "adv"
    )
    assert kind_rel(Path("lonely_file")) == Path("lonely")
