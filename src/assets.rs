//! Finding the files a model refers to (MTL libraries, textures) and turning texture bytes into
//! embeddable images.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::mesh::Image;

/// Where a reader finds the files a model refers to, and how it turns texture files into
/// images it can embed.
pub trait Assets {
    /// The file a model in `model_dir` refers to as `reference`: a relative path, a bare name,
    /// or another machine's absolute path.
    fn find(&self, reference: &str, model_dir: &Path) -> Option<PathBuf>;

    /// The bytes of a file [`find`](Self::find) returned, or of one a model names exactly (a
    /// glTF buffer, a USD sublayer). Reads the file by default.
    #[cfg(feature = "fs")]
    fn read(&self, path: &Path) -> Option<Cow<'_, [u8]>> {
        std::fs::read(path).ok().map(Cow::Owned)
    }

    /// The bytes of a file [`find`](Self::find) returned, or of one a model names exactly (a
    /// glTF buffer, a USD sublayer). Without the `fs` feature there is no default: the bytes
    /// come from wherever the assets keep them.
    #[cfg(not(feature = "fs"))]
    fn read(&self, path: &Path) -> Option<Cow<'_, [u8]>>;

    /// Texture bytes as an embeddable image; `name` is the file's name or path, for its format.
    /// By default only PNG and JPEG pass. Override to convert others (TGA, BMP, DDS, PSD, ...).
    fn image(&self, bytes: &[u8], name: &Path) -> Option<Image> {
        let _ = name;
        Image::png_or_jpeg(bytes)
    }

    /// A texture file read as an embeddable image.
    fn load_image(&self, path: &Path) -> Option<Image> {
        self.image(&self.read(path)?, path)
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
#[cfg(feature = "fs")]
#[derive(Debug, Clone)]
pub struct Folder {
    /// The top of the tree searched.
    pub root: PathBuf,
}

#[cfg(feature = "fs")]
impl Folder {
    /// A folder tree rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

#[cfg(feature = "fs")]
impl Assets for Folder {
    fn find(&self, reference: &str, model_dir: &Path) -> Option<PathBuf> {
        find_texture(reference, model_dir, &self.root)
    }

    fn color_texture_for(&self, model: &Path) -> Option<PathBuf> {
        color_texture_by_name(model, |visit| walk_images(&self.root, visit))
    }
}

/// Files held in memory by path: assets for loading without file access, found the way
/// [`Folder`] finds them on disk. Paths are relative to the top of the pack and `/`-separated
/// (`\` and a leading `./` or `/` read the same). A reference is tried next to the model, then
/// anywhere by file name, then as the same name in another image format; a model naming no
/// color texture takes one by name.
#[derive(Debug, Clone, Default)]
pub struct Memory {
    files: BTreeMap<String, Vec<u8>>,
}

impl Memory {
    /// No files.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) the file at `path`.
    pub fn insert(&mut self, path: &str, bytes: impl Into<Vec<u8>>) -> &mut Self {
        self.files.insert(memory_key(path), bytes.into());
        self
    }

    /// With the file at `path` added (or replaced).
    #[must_use]
    pub fn with(mut self, path: &str, bytes: impl Into<Vec<u8>>) -> Self {
        self.insert(path, bytes);
        self
    }

    /// The file at `path`.
    pub fn get(&self, path: &str) -> Option<&[u8]> {
        self.files.get(&memory_key(path)).map(Vec::as_slice)
    }

    /// The files' paths, normalized, in order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }

    fn images(&self, visit: &mut dyn FnMut(&Path, &str) -> bool) {
        for k in self.files.keys() {
            let p = Path::new(k);
            if let Some(stem) = image_stem(p)
                && visit(p, &stem)
            {
                return;
            }
        }
    }
}

impl Assets for Memory {
    fn find(&self, reference: &str, model_dir: &Path) -> Option<PathBuf> {
        let cleaned = reference.trim().replace('\\', "/");
        if cleaned.is_empty() {
            return None;
        }
        if !cleaned.starts_with('/') && !cleaned.contains(':') {
            let rel = memory_key(&format!("{}/{cleaned}", model_dir.to_string_lossy()));
            if self.files.contains_key(&rel) {
                return Some(rel.into());
            }
        }
        let name = cleaned.rsplit('/').next()?.to_lowercase();
        let by_name = self
            .files
            .keys()
            .find(|k| k.rsplit('/').next().is_some_and(|n| n.to_lowercase() == name));
        if let Some(k) = by_name {
            return Some(k.into());
        }
        same_name_other_format(&name, |visit| self.images(visit))
    }

    fn read(&self, path: &Path) -> Option<Cow<'_, [u8]>> {
        self.files
            .get(&memory_key(&path.to_string_lossy()))
            .map(|b| Cow::Borrowed(b.as_slice()))
    }

    fn color_texture_for(&self, model: &Path) -> Option<PathBuf> {
        color_texture_by_name(model, |visit| self.images(visit))
    }
}

/// A path as a [`Memory`] keys it: `/`-separated, without empty or `.` parts, and `..` stepping
/// up (no further than the top).
pub(crate) fn memory_key(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    parts.join("/")
}

/// A model's color texture by its name, from the images `walk` visits (until told to stop).
fn color_texture_by_name(
    model: &Path,
    walk: impl FnOnce(&mut dyn FnMut(&Path, &str) -> bool),
) -> Option<PathBuf> {
    let stem = model.file_stem()?.to_string_lossy().to_lowercase();
    if stem.len() < 3 || stem.contains('@') {
        return None;
    }
    // A name that says it's the color wins; one that says nothing ("Barrel.png") will do.
    let mut plain = None;
    let mut color = None;
    walk(&mut |p, file_stem| {
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

/// An image with the same name as `name` (lowercase) in another format (`rock.png` for
/// `rock.tga`, as packs often ship), preferring PNG and JPEG, from the images `walk` visits.
fn same_name_other_format(
    name: &str,
    walk: impl FnOnce(&mut dyn FnMut(&Path, &str) -> bool),
) -> Option<PathBuf> {
    let (stem, ext) = name.rsplit_once('.')?;
    if !IMAGE_EXTS.contains(&ext) {
        return None;
    }
    let mut best: Option<(u8, PathBuf)> = None;
    walk(&mut |p, s| {
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

/// An image file's lowercased stem; `None` for a file that isn't an image.
fn image_stem(p: &Path) -> Option<String> {
    let ext = p.extension()?.to_string_lossy().to_lowercase();
    IMAGE_EXTS.contains(&ext.as_str()).then(|| {
        p.file_stem()
            .map(|x| x.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    })
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

/// Visit the image files under `root` (at most 20 000 entries), with their lowercased stems,
/// until `visit` says stop.
#[cfg(feature = "fs")]
fn walk_images(root: &Path, visit: &mut dyn FnMut(&Path, &str) -> bool) {
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
            if let Some(stem) = image_stem(&p)
                && visit(&p, &stem)
            {
                return;
            }
        }
    }
}

/// Find a texture a model refers to: next to it, relative to it, or anywhere under `root` by
/// file name (artists' absolute paths like `C:\Users\me\tex\rock.tga` survive only as names).
/// Failing that, an image with the same name in another format.
#[cfg(feature = "fs")]
fn find_texture(reference: &str, model_dir: &Path, root: &Path) -> Option<PathBuf> {
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
    same_name_other_format(&name, |visit| walk_images(root, visit))
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;

    #[test]
    fn texture_roles_by_name() {
        assert_eq!(texture_role("barrel_normal"), Some(Role::Other));
        assert_eq!(texture_role("barrel_d"), Some(Role::Color));
        assert_eq!(texture_role("barrel_albedo"), Some(Role::Color));
        assert_eq!(texture_role("barrel"), None);
    }

    #[test]
    fn memory_finds_like_a_folder() {
        let m = Memory::new()
            .with("Models/crate.obj", "")
            .with("Models\\textures\\Crate.PNG", "png")
            .with("textures/rock.jpg", "jpg")
            .with("Barrel_Albedo.png", "a")
            .with("Barrel_Normal.png", "n");
        let found = |r: &str| {
            m.find(r, Path::new("Models"))
                .map(|p| p.to_string_lossy().into_owned())
        };
        let krate = Some("Models/textures/Crate.PNG");
        assert_eq!(found("textures/Crate.PNG").as_deref(), krate);
        assert_eq!(found("./textures\\Crate.PNG").as_deref(), krate);
        assert_eq!(found("D:\\Work\\crate.png").as_deref(), krate);
        assert_eq!(found("../textures/rock.jpg").as_deref(), Some("textures/rock.jpg"));
        assert_eq!(found("rock.tga").as_deref(), Some("textures/rock.jpg"));
        assert_eq!(found("missing.png"), None);
        assert_eq!(
            m.read(Path::new("./Models/textures/Crate.PNG")).as_deref(),
            Some(&b"png"[..])
        );
        assert_eq!(
            m.color_texture_for(Path::new("Barrel.obj")),
            Some(PathBuf::from("Barrel_Albedo.png"))
        );
    }
}
