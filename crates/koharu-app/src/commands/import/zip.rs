use std::{fs::File, io::Read, path::Path};

use anyhow::{Context as _, Result, bail};

use super::{EncodedPage, Format, MAX_PAGES, MAX_TOTAL_BYTES};

pub(super) fn extract(path: &Path) -> Result<Vec<EncodedPage>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open ZIP/CBZ container {}", path.display()))?;
    let mut archive = ::zip::ZipArchive::new(file)
        .with_context(|| format!("failed to read ZIP/CBZ container {}", path.display()))?;
    let mut images = Vec::new();
    let member_count = archive.len();
    let mut total: u64 = 0;

    for index in 0..member_count {
        let mut member = archive.by_index(index).with_context(|| {
            format!(
                "failed to read member {index} from ZIP/CBZ container {}",
                path.display()
            )
        })?;
        if member.is_dir() {
            continue;
        }
        let name = member.name().to_owned();
        let member_path = Path::new(&name);
        let supported = member_path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| matches!(extension.parse(), Ok(Format::Raster)));
        let metadata = member_path
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .is_none_or(|file_name| {
                file_name.starts_with("._")
                    || [".ds_store", "thumbs.db", "desktop.ini"]
                        .iter()
                        .any(|candidate| file_name.eq_ignore_ascii_case(candidate))
            })
            || member_path.components().any(|component| {
                component
                    .as_os_str()
                    .to_str()
                    .is_some_and(|component| component.eq_ignore_ascii_case("__macosx"))
            });
        if !supported || metadata {
            continue;
        }
        if images.len() >= MAX_PAGES {
            bail!(
                "ZIP/CBZ container {} contains more than {MAX_PAGES} image pages",
                path.display()
            );
        }
        // Bounded take guards against decompression bombs: each member may
        // consume at most the remaining 512 MiB budget, no matter what its
        // header declares.
        let limit = MAX_TOTAL_BYTES - total;
        let mut bytes = Vec::new();
        member
            .by_ref()
            .take(limit + 1)
            .read_to_end(&mut bytes)
            .with_context(|| {
                format!(
                    "failed to read image member {name:?} from ZIP/CBZ container {}",
                    path.display()
                )
            })?;
        if bytes.len() as u64 > limit {
            bail!(
                "ZIP/CBZ container {} exceeds the {MAX_TOTAL_BYTES}-byte decompressed import limit at member {name:?}",
                path.display()
            );
        }
        total += bytes.len() as u64;
        images.push(EncodedPage { name, bytes });
    }

    alphanumeric_sort::sort_slice_by_os_str_key(&mut images, |image| {
        std::ffi::OsStr::new(&image.name)
    });
    if images.is_empty() {
        bail!(
            "ZIP/CBZ container {} contains no supported raster images",
            path.display()
        );
    }
    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageFormat;
    use std::{
        io::{Cursor, Write},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn sample_png() -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(2, 3, image::Rgba([255, 0, 0, 255]));
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .expect("encode PNG");
        bytes.into_inner()
    }

    fn archive_path(suffix: &str) -> PathBuf {
        // Parallel tests share the process id, and the wall clock is too coarse
        // on Windows to keep their archives apart.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "koharu-import-{}-{timestamp}-{index}.{suffix}",
            std::process::id()
        ))
    }

    fn write_archive(entries: &[(&str, &[u8])]) -> PathBuf {
        let path = archive_path("cbz");
        let file = File::create(&path).expect("create archive");
        let mut archive = ::zip::ZipWriter::new(file);
        let options = ::zip::write::SimpleFileOptions::default();
        for (name, bytes) in entries {
            archive.start_file(*name, options).expect("start member");
            archive.write_all(bytes).expect("write member");
        }
        archive.finish().expect("finish archive");
        path
    }

    #[test]
    fn reads_images_in_natural_member_order_and_skips_metadata() {
        let png = sample_png();
        let path = write_archive(&[
            ("ComicInfo.xml", b"<ComicInfo />"),
            ("pages/page10.png", &png),
            ("pages/page2.png", &png),
            ("__MACOSX/._page3.png", &png),
        ]);
        let images = extract(&path).expect("read archive");
        std::fs::remove_file(&path).expect("remove archive");
        assert_eq!(
            images
                .iter()
                .map(|image| image.name.as_str())
                .collect::<Vec<_>>(),
            ["pages/page2.png", "pages/page10.png"]
        );
    }

    #[test]
    fn rejects_archive_without_supported_images() {
        let path = write_archive(&[("ComicInfo.xml", b"<ComicInfo />")]);
        let error = extract(&path).expect_err("archive should be rejected");
        std::fs::remove_file(&path).expect("remove archive");
        assert!(error.to_string().contains("no supported raster images"));
    }

    #[test]
    fn rejects_more_pages_than_the_import_limit() {
        let png = sample_png();
        let path = archive_path("cbz");
        let file = File::create(&path).expect("create archive");
        let mut archive = ::zip::ZipWriter::new(file);
        let options = ::zip::write::SimpleFileOptions::default();
        for index in 0..MAX_PAGES + 1 {
            archive
                .start_file(format!("page{index:04}.png"), options)
                .expect("start member");
            archive.write_all(&png).expect("write member");
        }
        archive.finish().expect("finish archive");

        let error = extract(&path).expect_err("archive should be rejected");
        std::fs::remove_file(&path).expect("remove archive");
        assert!(error.to_string().contains("image pages"));
    }

    #[test]
    fn rejects_member_beyond_the_decompressed_byte_limit() {
        let path = archive_path("cbz");
        let file = File::create(&path).expect("create archive");
        let mut archive = ::zip::ZipWriter::new(file);
        let options = ::zip::write::SimpleFileOptions::default()
            .compression_method(::zip::CompressionMethod::Deflated)
            .compression_level(Some(1));
        archive
            .start_file("page.png", options)
            .expect("start member");
        let mut zeros = std::io::repeat(0).take(MAX_TOTAL_BYTES + 1);
        std::io::copy(&mut zeros, &mut archive).expect("write member");
        archive.finish().expect("finish archive");

        let error = extract(&path).expect_err("archive should be rejected");
        std::fs::remove_file(&path).expect("remove archive");
        assert!(error.to_string().contains("decompressed import limit"));
    }
}
