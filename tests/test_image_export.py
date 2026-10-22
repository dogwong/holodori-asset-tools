import unittest
from types import SimpleNamespace
from unittest.mock import patch
import struct

from PIL import Image

from holodori_asset_tools.entrypoint.extract import image_for_export
from holodori_asset_tools.image_sizes import restore_sprite_canvas, LAYOUTS, _scaled_sprite_image, image_output_paths, texture_canvas_sizes, texture_image


class ImageExportTests(unittest.TestCase):
    def test_full_card_uses_ui_reference_canvas(self):
        image = Image.new("RGBA", (2048, 1024), (12, 34, 56, 128))
        result = image_for_export("Texture2D", "img_card_full_example", image)
        self.assertEqual(result.size, (1920, 1080))
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getpixel((960, 540))[3], 128)
        self.assertEqual(image.size, (2048, 1024))

    def test_unverified_images_keep_their_pixels(self):
        for kind, name, size in [
            ("Sprite", "spr_env_monitor_example", (336, 24)),
            ("Sprite", "img_card_full_example", (336, 24)),
            ("Texture2D", "img_unknown_example", (2048, 1024)),
            ("Texture2D", "img_gacha_group_icon_example", (1024, 256)),
            ("Texture2D", "img_card_thumb_example", (512, 256)),
            ("Texture2D", "img_card_full_example", (1024, 2048)),
            ("Texture2D", "img_card_full_example", (1920, 1080)),
        ]:
            with self.subTest(kind=kind, name=name, size=size):
                image = Image.new("RGBA", size, (12, 34, 56, 128))
                self.assertIs(image_for_export(kind, name, image), image)

    def test_different_categories_use_different_layouts(self):
        examples = [
            ("img_gacha_group_icon_1", (256, 128), (256, 160)),
            ("img_card_vert_1", (256, 512), (358, 512)),
            ("img_card_thin_1", (512, 2048), (585, 2048)),
            ("img_work_head_1", (2048, 128), (2048, 97)),
            ("img_mv_slide_1", (2048, 1024), (2048, 1152)),
            ("img_ui_tips-game_1", (1024, 512), (1024, 576)),
            ("img_marathon_event_normal_bg-001-01", (2048, 1024), (1920, 1080)),
        ]
        for name, source, target in examples:
            with self.subTest(name=name):
                self.assertEqual(image_for_export("Texture2D", name, Image.new("RGBA", source)).size, target)

    def test_aspect_rounding_is_within_half_a_pixel(self):
        for profile in LAYOUTS:
            if profile.exact_reference:
                continue
            for stored in profile.stored_sizes:
                result = profile.output_size(stored)
                scale = max(stored) / max(profile.reference)
                self.assertEqual(max(stored), max(result))
                for actual, reference in zip(result, profile.reference):
                    self.assertLessEqual(abs(actual - reference * scale), 0.5)

    def test_sprite_padding_preserves_content_alpha_and_bottom_left_offset(self):
        content = Image.new("RGBA", (3, 2), (12, 34, 56, 128))
        content.putpixel((0, 0), (255, 0, 0, 255))
        result = restore_sprite_canvas(content, (8, 7), (2.1, 1.1))
        self.assertEqual(result.size, (8, 7))
        self.assertEqual(result.getpixel((2, 4)), (255, 0, 0, 255))
        self.assertEqual(result.getpixel((4, 5)), (12, 34, 56, 128))
        self.assertEqual(result.getpixel((2, 3)), (0, 0, 0, 0))
        self.assertEqual(result.getpixel((2, 6)), (0, 0, 0, 0))
        self.assertIs(restore_sprite_canvas(content, (3, 2), (0, 0)), content)

    def test_invalid_sprite_bounds_are_not_silently_cropped(self):
        with self.assertRaises(ValueError):
            restore_sprite_canvas(Image.new("RGBA", (4, 4)), (3, 3), (0, 0))

    def test_scaled_atlas_crops_in_logical_coordinates(self):
        # A 2x2 storage atlas represents a 4x4 logical atlas. The rightmost
        # logical column would be out of bounds without multiplier handling.
        sprite = SimpleNamespace(assets_file=SimpleNamespace(_cache={}))
        data = SimpleNamespace(settingsRaw=2, texture=SimpleNamespace(file_id=0, path_id=1),
                               alphaTexture=None, textureRect=SimpleNamespace(x=3, y=0, width=1, height=4))
        native = Image.new("RGBA", (2, 2), (12, 34, 56, 128))
        with patch('UnityPy.export.SpriteHelper.get_image', return_value=native):
            result = _scaled_sprite_image(sprite, data, 0.5)
        self.assertEqual(result.size, (1, 4))
        self.assertEqual(result.getchannel('A').getextrema(), (128, 128))

    def test_same_name_and_repeated_path_ids_do_not_overwrite(self):
        records = [dict(serialized_file='A', id=1, name='console', size=(32, 32)),
                   dict(serialized_file='B', id=1, name='console', size=(64, 64)),
                   dict(serialized_file='A', id=2, name='console', size=(32, 32))]
        outputs = image_output_paths(records)
        self.assertEqual(str(outputs[('B', 1)]), 'console.png')
        self.assertEqual(len(set(outputs.values())), 3)
        self.assertEqual(image_output_paths(list(reversed(records))), outputs)

    def test_explicit_canvas_requires_local_full_uv_reference(self):
        def obj(kind, path_id, value):
            return SimpleNamespace(type=SimpleNamespace(name=kind), assets_file=SimpleNamespace(name='A'),
                                   path_id=path_id, read=lambda: SimpleNamespace(m_ClassName=value),
                                   read_typetree=lambda: value)
        def ptr(path_id, file_id=0):
            return dict(m_FileID=file_id, m_PathID=path_id)
        raw = dict(m_GameObject=ptr(10), m_Script=ptr(2), m_Texture=ptr(99),
                   m_UVRect=dict(x=0, y=0, width=1, height=1))
        env = SimpleNamespace(objects=[obj('MonoScript', 1, 'ClippingRectFitter'), obj('MonoScript', 2, 'RawImage'),
            obj('MonoBehaviour', 3, dict(m_GameObject=ptr(10), m_Script=ptr(1), _baseSize=dict(x=4160, y=2340))),
            obj('MonoBehaviour', 4, raw)])
        self.assertEqual(texture_canvas_sizes(env), {('A', 99): (4160, 2340)})
        raw['m_UVRect']['width'] = 0.5
        self.assertEqual(texture_canvas_sizes(env), {})
        raw['m_UVRect']['width'] = 1
        raw['m_Texture']['m_FileID'] = 1
        self.assertEqual(texture_canvas_sizes(env), {})

    def test_half_float_png_handles_one_negative_and_hdr_values(self):
        from UnityPy.enums import TextureFormat
        texture = SimpleNamespace(m_TextureFormat=TextureFormat.RGBAHalf, m_Width=1, m_Height=1,
                                  m_Name='pose', get_image_data=lambda: struct.pack('<4e', -1, 0.5, 2, 1))
        image = texture_image(texture)
        self.assertEqual(image.mode, 'RGBA')
        self.assertEqual(image.getpixel((0, 0)), (0, 128, 255, 255))


if __name__ == "__main__":
    unittest.main()
