//! Finding the files a model refers to (MTL libraries, textures) and turning texture bytes into
//! embeddable images.

use std::path::{Path, PathBuf};

use crate::mesh::Image;

/// Where a reader finds the files a model refers to, and how it turns texture files into
/// images it can embed.
pub trait Assets {
    /// The file a model in `model_dir` refers to as `reference`: a relative path, a bare name,
    /// or another machine's absolute path.
    fn find(&self, reference: &str, model_dir: &Path) -> Option<PathBuf>;

    /// Texture bytes as an embeddable image; `name` is the file's name or path, for its format.
    /// By default only PNG and JPEG pass. Override to convert others (TGA, BMP, DDS, PSD, ...).
    fn image(&self, bytes: &[u8], name: &Path) -> Option<Image> {
        let _ = name;
        Image::png_or_jpeg(bytes)
    }

    /// A texture file read as an embeddable image.
    fn load_image(&self, path: &Path) -> Option<Image> {
        self.image(&std::fs::read(path).ok()?, path)
    }

    /// A color texture for a model at `model` that names none, by naming convention. `None` by
    /// default.
    fn color_texture_for(&self, model: &Path) -> Option<PathBuf> {
        let _ = model;
        None
    }
}

/// Assets in a folder tree laid out like the pack a model came in: references are tried next to
/// the model, then found anywhere under `root` by file name (or the same name in another image
/// format). A model naming no color texture takes one by name: `Barrel.png` or
/// `Barrel_Albedo.png` for `Barrel.obj`.
#[derive(Debug, Clone)]
pub struct Folder {
    /// The top of the tree searched.
    pub root: PathBuf,
}

impl Folder {
    /// A folder tree rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Assets for Folder {
    fn find(&self, reference: &str, model_dir: &Path) -> Option<PathBuf> {
        find_texture(reference, model_dir, &self.root)
    }

    fn color_texture_for(&self, model: &Path) -> Option<PathBuf> {
        let stem = model.file_stem()?.to_string_lossy().to_lowercase();
        if stem.len() < 3 || stem.contains('@') {
            return None;
        }
        // A name that says it's the color wins; one that says nothing ("Barrel.png") will do.
        let mut plain = None;
        let mut color = None;
        walk_images(&self.root, |p, file_stem| {
            let Some(rest) = file_stem.strip_prefix(&stem) else {
                return false;
            };
            if !(rest.is_empty() || rest.starts_with(['_', '-', ' ', '.'])) {
                return false;
            }
            match texture_role(file_stem) {
                Some(Role::Color) if color.is_none() => color = Some(p.to_path_buf()),
                None if plain.is_none() => plain = Some(p.to_path_buf()),
                _ => {}
            }
            color.is_some()
        });
        color.or(plain)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Role {
    Color,
    Other,
}

/// What a texture's name says it holds: `None` when it says nothing.
fn texture_role(stem: &str) -> Option<Role> {
    let tail = stem.rsplit(['-', '_', '.', ' ']).next().unwrap_or(stem);
    let split = tail.len() < stem.len();
    let ends = |list: &[&str]| split && list.contains(&tail);
    let has = |list: &[&str]| list.iter().any(|w| stem.contains(w));
    if has(&[
        "normal",
        "nrm",
        "emissi",
        "emit",
        "glow",
        "rough",
        "metal",
        "occlusion",
        "height",
        "disp",
        "bump",
    ]) || has(&["mask", "opacity", "alpha", "spec", "gloss", "smooth"])
        || ends(&[
            "n", "nor", "norm", "e", "r", "m", "mt", "ao", "occ", "h", "orm", "arm", "mra", "rma",
            "s", "g",
        ])
    {
        return Some(Role::Other);
    }
    if has(&[
        "albedo",
        "basecolor",
        "base_color",
        "base color",
        "basecolour",
        "diffuse",
    ]) || ends(&["diff", "dif", "col", "color", "colour", "d", "bc", "a", "c"])
    {
        return Some(Role::Color);
    }
    None
}

const IMAGE_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "tga", "bmp", "webp", "gif", "tif", "tiff", "dds", "psd",
];

/// Visit the image files under `root` (at most 20 000 entries), with their lowercased stems, until `visit` says
/// stop.
fn walk_images(root: &Path, mut visit: impl FnMut(&Path, &str) -> bool) {
    let mut stack = vec![root.to_path_buf()];
    let mut seen = 0;
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            seen += 1;
            if seen > 20_000 {
                return;
            }
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let ext = p
                .extension()
                .map(|x| x.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if !IMAGE_EXTS.contains(&ext.as_str()) {
                continue;
            }
            let stem = p
                .file_stem()
                .map(|x| x.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if visit(&p, &stem) {
                return;
            }
        }
    }
}

/// Find a texture a model refers to: next to it, relative to it, or anywhere under `root` by file name (artists'
/// absolute paths like `C:\Users\me\tex\rock.tga` survive only as names). Failing that, an image with the same
/// name in another format (`rock.png` for `rock.tga`, as packs often ship), preferring PNG and JPEG.
fn find_texture(reference: &str, model_dir: &Path, root: &Path) -> Option<std::path::PathBuf> {
    let cleaned = reference.trim().replace('\\', "/");
    if cleaned.is_empty() {
        return None;
    }
    let rel = model_dir.join(&cleaned);
    if !cleaned.starts_with('/')
        && !cleaned.contains(':')
        && !cleaned.split('/').any(|c| c == "..")
        && rel.is_file()
    {
        return Some(rel);
    }
    let name = cleaned.rsplit('/').next()?.to_lowercase();
    let mut stack = vec![root.to_path_buf()];
    let mut seen = 0;
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            seen += 1;
            if seen > 20_000 {
                return None;
            }
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .file_name()
                .is_some_and(|n| n.to_string_lossy().to_lowercase() == name)
            {
                return Some(p);
            }
        }
    }
    let is_image = name
        .rsplit_once('.')
        .is_some_and(|(_, e)| IMAGE_EXTS.contains(&e));
    let stem = name.rsplit_once('.').map_or(name.as_str(), |(s, _)| s);
    if !is_image {
        return None;
    }
    let mut best: Option<(u8, std::path::PathBuf)> = None;
    walk_images(root, |p, s| {
        if s == stem {
            let ext = p
                .extension()
                .map(|x| x.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let rank = u8::from(!matches!(ext.as_str(), "png" | "jpg" | "jpeg"));
            if best.as_ref().is_none_or(|(r, _)| rank < *r) {
                best = Some((rank, p.to_path_buf()));
            }
        }
        best.as_ref().is_some_and(|(r, _)| *r == 0)
    });
    best.map(|(_, p)| p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_roles_by_name() {
        assert_eq!(texture_role("barrel_normal"), Some(Role::Other));
        assert_eq!(texture_role("barrel_d"), Some(Role::Color));
        assert_eq!(texture_role("barrel_albedo"), Some(Role::Color));
        assert_eq!(texture_role("barrel"), None);
    }
}
