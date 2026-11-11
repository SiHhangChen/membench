"""Task-local milk packaging texture recoloring."""

import mujoco
import numpy as np

from .domain.contracts import PACKAGING_RGB


class CartonAppearance:
    def _apply_carton_material(self, item, packaging) -> None:
        """Recolor the milk texture and apply its matching optical finish."""
        profile = {
            "texture_rgb": PACKAGING_RGB[packaging],
            "specular": 0.28,
            "shininess": 0.25,
            "reflectance": 0.0,
        }
        model = self.sim.model
        root = int(self.obj_body_id[item])
        parents = np.asarray(model.body_parentid, dtype=int)
        material_ids: set[int] = set()
        for geom_id, body_id in enumerate(np.asarray(model.geom_bodyid, dtype=int)):
            current = int(body_id)
            belongs = False
            while current > 0:
                if current == root:
                    belongs = True
                    break
                current = int(parents[current])
            if not belongs or float(model.geom_rgba[geom_id, 3]) <= 0.0:
                continue
            material_id = int(model.geom_matid[geom_id])
            if material_id >= 0:
                material_ids.add(material_id)
            # A neutral multiplier displays the recolored texture directly.
            model.geom_rgba[geom_id, :] = (1.0, 1.0, 1.0, 1.0)
        for material_id in material_ids:
            model.mat_specular[material_id] = float(profile["specular"])
            model.mat_shininess[material_id] = float(profile["shininess"])
            model.mat_reflectance[material_id] = float(profile["reflectance"])

        raw_model = model._model
        texture_ids = {
            int(texture_id)
            for material_id in material_ids
            for texture_id in np.asarray(raw_model.mat_texid[material_id]).reshape(-1)
            if int(texture_id) >= 0
        }
        cache = getattr(self, "_carton_texture_original_data", None)
        if cache is None:
            cache = {}
            self._carton_texture_original_data = cache
        target = np.asarray(profile["texture_rgb"], dtype=np.float32)
        source_green = np.asarray((0, 107, 14), dtype=np.float32)
        white = np.full(3, 255.0, dtype=np.float32)
        green_axis = source_green - white
        green_axis_norm = float(np.dot(green_axis, green_axis))
        for texture_id in texture_ids:
            address = int(raw_model.tex_adr[texture_id])
            width = int(raw_model.tex_width[texture_id])
            height = int(raw_model.tex_height[texture_id])
            channels = int(raw_model.tex_nchannel[texture_id])
            size = width * height * channels
            original = cache.setdefault(
                texture_id,
                np.asarray(raw_model.tex_data[address : address + size], dtype=np.uint8).copy(),
            )
            pixels = original.reshape(height, width, channels).copy()
            rgb = pixels[..., :3].astype(np.float32)
            green_mask = (rgb[..., 1] > rgb[..., 0] + 2.0) & (rgb[..., 1] > rgb[..., 2] + 2.0)
            coverage = np.sum((rgb - white) * green_axis, axis=-1) / green_axis_norm
            coverage = np.clip(coverage, 0.0, 1.0)
            recolored = rgb + coverage[..., None] * (target - source_green)
            pixels[..., :3][green_mask] = np.clip(recolored[green_mask], 0.0, 255.0).astype(
                np.uint8
            )
            raw_model.tex_data[address : address + size] = pixels.reshape(-1)
            self._upload_texture_if_rendering(texture_id)

    def _upload_texture_if_rendering(self, texture_id: int) -> None:
        """Refresh an existing GPU texture after changing model.tex_data."""
        context = getattr(self.sim, "_render_context_offscreen", None)
        if context is None:
            return
        context.gl_ctx.make_current()
        mujoco.mjr_uploadTexture(self.sim.model._model, context.con, int(texture_id))

    def _set_body_color(self, name: str, rgba: tuple[float, float, float, float]) -> None:
        model = self.sim.model
        root = int(self.obj_body_id[name])
        parents = np.asarray(model.body_parentid, dtype=int)
        for geom_id, body_id in enumerate(np.asarray(model.geom_bodyid, dtype=int)):
            current = int(body_id)
            belongs = False
            while current > 0:
                if current == root:
                    belongs = True
                    break
                current = int(parents[current])
            if belongs and float(model.geom_rgba[geom_id, 3]) > 0.0:
                model.geom_rgba[geom_id, :3] = np.asarray(rgba[:3], dtype=float)
