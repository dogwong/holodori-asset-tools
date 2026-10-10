from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from hashlib import sha256
from math import isfinite
from logging import getLogger
from pathlib import Path
import struct

from PIL import Image

logger = getLogger("extract")


def texture_image(texture) -> Image.Image:
    """Decode RGBAHalf as a clamped PNG preview (the bundle retains HDR values).

    UnityPy 1.25's half-float converter multiplies by 256 and can raise on 1.0,
    negative values, or HDR values. The client's audience pose textures use it.
    """
    from UnityPy.enums import TextureFormat

    if texture.m_TextureFormat != TextureFormat.RGBAHalf:
        return texture.image
    size = (texture.m_Width, texture.m_Height)
    count = size[0] * size[1] * 4
    data = texture.get_image_data()[:count * 2]
    if len(data) != count * 2:
        raise ValueError('Truncated RGBAHalf texture')
    pixels = bytes(round(min(1.0, max(0.0, value)) * 255) if isfinite(value) else (255 if value > 0 else 0)
                   for (value,) in struct.iter_unpack('<e', data))
    logger.info('%s: RGBAHalf exported as clamped 8-bit PNG preview', texture.m_Name)
    return Image.frombytes('RGBA', size, pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def texture_canvas_sizes(env) -> dict[tuple[str, int], tuple[int, int]]:
    """Read explicit ClippingRectFitter canvases linked to same-bundle RawImages.

    The full base canvas is exported, not the visible clipping rectangle.
    Require direct local references and full UVs; never infer from a parent size.
    """
    scripts = {(o.assets_file.name, o.path_id): o.read().m_ClassName
               for o in env.objects if o.type.name == 'MonoScript'}
    if 'ClippingRectFitter' not in scripts.values():
        return {}
    fitters = {}
    images = []
    for obj in env.objects:
        if obj.type.name != 'MonoBehaviour':
            continue
        data = obj.read_typetree()
        script = data.get('m_Script', {})
        if script.get('m_FileID') != 0:
            continue
        cls = scripts.get((obj.assets_file.name, script.get('m_PathID')))
        go = data.get('m_GameObject', {})
        if go.get('m_FileID') != 0:
            continue
        key = (obj.assets_file.name, go.get('m_PathID'))
        if cls == 'ClippingRectFitter':
            base = data.get('_baseSize', {})
            size = (base.get('x', 0), base.get('y', 0))
            if all(isfinite(v) and 0 < v <= 32768 for v in size):
                fitters[key] = (round(size[0]), round(size[1]))
        elif cls in ('RawImage', 'VisionRawImage'):
            texture = data.get('m_Texture', {})
            if texture.get('m_FileID') == 0 and texture.get('m_PathID') and data.get('m_UVRect') == dict(x=0, y=0, width=1, height=1):
                images.append((key, (obj.assets_file.name, texture['m_PathID'])))
    sizes = defaultdict(set)
    for go, texture in images:
        if go in fitters:
            sizes[texture].add(fitters[go])
    return {texture: next(iter(values)) for texture, values in sizes.items() if len(values) == 1}


def image_output_paths(records) -> dict[tuple[str, int], Path]:
    """Keep the largest same-name image at its familiar path; retain other objects."""
    groups = defaultdict(list)
    for row in records:
        groups[Path((row['name'] or str(row['id'])) + '.png')].append(row)
    result = {}
    for path, group in groups.items():
        ordered = sorted(group, key=lambda r: (-r['size'][0] * r['size'][1], r['serialized_file'], r['id']))
        for index, row in enumerate(ordered):
            dest = path
            if index:
                file_id = sha256(row['serialized_file'].encode()).hexdigest()[:8]
                dest = path.with_name(f'{path.stem}__{file_id}_{row["id"]}.png')
            result[(row['serialized_file'], row['id'])] = dest
    return result


@dataclass(frozen=True)
class ImageLayout:
    prefix: str
    stored_sizes: tuple[tuple[int, int], ...]
    reference: tuple[int, int]
    binding: str
    exact_reference: bool = False

    def output_size(self, stored: tuple[int, int]) -> tuple[int, int]:
        if self.exact_reference:
            return self.reference
        width, height = self.reference
        # Preserve the source's longest dimension, with integer-pixel rounding.
        scale = max(stored) / max(width, height)
        return max(1, round(width * scale)), max(1, round(height * scale))


# A name AND a verified storage size must match before any resampling occurs.
LAYOUTS = (
    ImageLayout("img_card_full_", ((2048, 1024),), (1920, 1080), "CardZoomScreenView", True),
    ImageLayout("img_card_vert_", ((256, 512),), (224, 320), "VerticalCardThumbnail._commonView"),
    ImageLayout("img_card_thin_", ((256, 1024), (512, 2048)), (160, 560), "GachaResultCardView._cardThumbnail"),
    ImageLayout("img_banner_full_", ((2048, 1024),), (384, 216), "NoticeBannerListCellView._bannerImage"),
    ImageLayout("img_gacha_group_icon_", ((256, 128),), (208, 130), "GachaGroupListCellView._groupIcon"),
    ImageLayout("img_gacha_top_", ((2048, 1024),), (16, 9), "GachaBackgroundView._promotionImageView"),
    ImageLayout("img_chr_key_band_", ((128, 512),), (107, 512), "CharacterKeyView._keyBandTextureSetter"),
    ImageLayout("img_chr_key_chain_", ((128, 256),), (60, 128), "CharacterKeyView._keyChainTextureSetter"),
    ImageLayout("img_chr_mem_thumb_", ((512, 1024),), (290, 742), "MembershipTopCharacterListCellView._characterImage"),
    ImageLayout("img_chr_room_", ((2048, 1024),), (16, 9), "CharacterDetailScreenView._roomImageSetter"),
    ImageLayout("img_work_bg_", ((2048, 1024),), (16, 9), "WorkBackImageSetter._backGroundImageTextureSetter"),
    ImageLayout("img_work_head_", ((2048, 128),), (4096, 194), "WorkBackImageSetter._headerImageTextureSetter"),
    ImageLayout("img_work_thumb_", ((512, 256),), (480, 280), "WorkSlotDetailView._thumbnail"),
    ImageLayout("img_ui_tips", ((1024, 512),), (704, 396), "TutorialTipsDialogListCellView._tipsImageSetter"),
    ImageLayout("img_ui_loading_", ((1024, 512),), (936, 528), "VisionLoadingImageTipsListCellView._image"),
    ImageLayout("img_login_logo_", ((1024, 512),), (616, 385), "NormalLoginBonusScreenView._logoImage"),
    ImageLayout("img_exchange_bg_", ((2048, 1024),), (16, 9), "ExchangeTopScreenView._backGroundImageTextureSetter"),
    ImageLayout("img_exchange_head_", ((2048, 256),), (2340, 346), "ExchangeTopScreenView._headerImageTextureSetter"),
    ImageLayout("img_story_chapter_", ((2048, 1024),), (464, 261), "StoryArchiveStorySelectTabButton._rawImageTextureSetter"),
    ImageLayout("img_mg_logo_", ((1024, 512),), (576, 360), "MiniGameTopScreenView._logoImage"),
    ImageLayout("img_mg_thumb_", ((1024, 512),), (16, 9), "MiniGameTopVideoPlayer._thumbnail"),
    ImageLayout("img_mv_slide_", ((2048, 1024), (1024, 512)), (1920, 1080), "SlideMvBackgroundView._rawImage1"),
    ImageLayout("img_poster_full_", ((2048, 1024),), (256, 144), "CustomPaletteBaseDecorationPosterListCellView._posterSetter"),
    ImageLayout("img_startup_full_", ((2048, 1024),), (1344, 756), "StartupContentCoverView._startupNotificationImage"),
    ImageLayout("img_ui_title_", ((2048, 1024),), (16, 9), "TitleSceneView._bgTextureSetter"),
    ImageLayout("img_gacha_logo_", ((256, 128),), (200, 136), "GachaLogoListCellView._logoImage"),
    ImageLayout("img_fishing_fish_detail_", ((1024, 512),), (702, 344), "ParkFishingPictureBookScreenView._fishImageSetter"),
    ImageLayout("img_login_bg_", ((2048, 1024),), (2538, 1440), "EventLoginBonusScreenView._backgroundImage"),
    ImageLayout("img_acc_shop_", ((512, 1024),), (440, 600), "CharacterItemShopSelectItemCell._thumbnailSetter"),
    ImageLayout("img_hair_shop_", ((512, 1024),), (440, 600), "CharacterItemShopSelectItemCell._thumbnailSetter"),
    # ResourceDetail's size-switch values; other screens use slightly different ratios.
    ImageLayout("img_emb_m_", ((1024, 512),), (578, 271), "UserEmblemCellView._cellSizeChange"),
    ImageLayout("img_emb_l_", ((1024, 256),), (884, 272), "UserEmblemCellView._cellSizeChange"),
)


def image_layout(name: str, size: tuple[int, int]) -> ImageLayout | None:
    return next((p for p in LAYOUTS if name.startswith(p.prefix) and size in p.stored_sizes), None)


def image_for_export(kind: str, name: str, image: Image.Image) -> Image.Image:
    if kind != "Texture2D":
        return image
    profile = image_layout(name, image.size)
    if profile is None:
        for candidate in LAYOUTS:
            if name.startswith(candidate.prefix) and image.size != candidate.output_size(candidate.stored_sizes[0]):
                logger.warning("%s: unverified texture size %s; keeping native size", name, image.size)
                break
        return image
    target = profile.output_size(image.size)
    return image if target == image.size else image.resize(target, Image.Resampling.LANCZOS)


def restore_sprite_canvas(image: Image.Image, size: tuple[int, int], offset: tuple[float, float]) -> Image.Image:
    """Place decoded, unrotated content on its logical canvas without stretching."""
    x = round(offset[0])
    y = size[1] - round(offset[1]) - image.height
    if image.size == size and x == 0 and y == 0:
        return image
    if min(size) <= 0 or x < 0 or y < 0 or x + image.width > size[0] or y + image.height > size[1]:
        raise ValueError(f"Sprite content {image.size} at {(x, y)} exceeds canvas {size}")
    canvas = Image.new("RGBA", size)
    # No mask: using the source alpha as a mask would apply alpha twice.
    canvas.paste(image.convert("RGBA"), (x, y))
    return canvas


def sprite_for_export(sprite) -> Image.Image:
    """UnityPy handles packed rotation/mesh; restore the trimmed outer margins."""
    atlas = None
    if sprite.m_SpriteAtlas:
        atlas = sprite.m_SpriteAtlas.deref_parse_as_object()
    elif sprite.m_AtlasTags:
        for obj in sprite.assets_file.objects.values():
            if obj.type.name == "SpriteAtlas" and obj.peek_name() == sprite.m_AtlasTags[0]:
                atlas = obj.read()
                break
    data = (next(value for key, value in atlas.m_RenderDataMap if key == sprite.m_RenderDataKey)
            if atlas else sprite.m_RD)
    multiplier = getattr(data, "downscaleMultiplier", 1.0)
    image = (_scaled_sprite_image(sprite, data, multiplier)
             if multiplier and multiplier != 1.0 else sprite.image)
    rect = sprite.m_Rect
    offset = data.textureRectOffset
    return restore_sprite_canvas(image, (round(rect.width), round(rect.height)), (offset.x, offset.y))


def _scaled_sprite_image(sprite, data, multiplier: float) -> Image.Image:
    """UnityPy 1.25 omits atlas downscaleMultiplier in the crop-only path.

    Restore the atlas coordinate space before cropping, as AssetStudio does.
    UV-based meshes already render into logical geometry coordinates.
    """
    from UnityPy.export import SpriteHelper as helper

    if multiplier <= 0:
        raise ValueError(f"Invalid atlas downscale multiplier: {multiplier}")
    settings = helper.SpriteSettings(data.settingsRaw)
    mesh = None
    if settings.packingMode == helper.SpritePackingMode.kSPMTight:
        mesh = helper.MeshHandler(sprite.m_RD, sprite.object_reader.version)
        mesh.process()
        if mesh.m_UV0 and any(u or v for u, v in mesh.m_UV0):
            return sprite.image
    cache = sprite.assets_file._cache
    key = ("holodori_scaled_atlas", data.texture.file_id, data.texture.path_id,
           data.alphaTexture.file_id if data.alphaTexture else 0,
           data.alphaTexture.path_id if data.alphaTexture else 0, multiplier)
    if key not in cache:
        original = helper.get_image(sprite, data.texture, data.alphaTexture)
        cache[key] = original.resize((round(original.width / multiplier), round(original.height / multiplier)),
                                     Image.Resampling.LANCZOS)
    rect = data.textureRect
    image = cache[key].crop((rect.x, rect.y, rect.x + rect.width, rect.y + rect.height))
    rotations = {
        helper.SpritePackingRotation.kSPRFlipHorizontal: Image.Transpose.FLIP_LEFT_RIGHT,
        helper.SpritePackingRotation.kSPRFlipVertical: Image.Transpose.FLIP_TOP_BOTTOM,
        helper.SpritePackingRotation.kSPRRotate180: Image.Transpose.ROTATE_180,
        helper.SpritePackingRotation.kSPRRotate90: Image.Transpose.ROTATE_270,
    }
    if settings.packed and settings.packingRotation in rotations:
        image = image.transpose(rotations[settings.packingRotation])
    if mesh is not None:
        image = helper.mask_sprite(sprite, mesh, image)
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
