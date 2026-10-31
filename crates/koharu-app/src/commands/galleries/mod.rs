mod http;
mod sources;

use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result, bail};
use koharu_secrets::ExposeSecret as _;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{AppHandle, Cef, Manager as _, ipc::Channel};
use unicode_normalization::UnicodeNormalization as _;
use uuid::Uuid;

use self::sources::{Inspection, Sources};
use super::{
    Error,
    import::MAX_TOTAL_BYTES,
    lifecycle::{
        decode_import_paths, ensure_page_import_allowed, import_decoded_pages, replace_project,
    },
    project::ProjectLibrary,
};

const MAX_GALLERIES: usize = 20;
const MAX_PAGES: usize = 500;
const EXHENTAI_COOKIES_KEY: &str = "gallery_exhentai_cookies";

#[derive(Clone, Debug, Deserialize, Type)]
pub struct GalleryImportRequest {
    pub urls: Vec<String>,
    pub name: Option<String>,
    pub exhentai_cookies: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum GalleryImportStatus {
    Inspecting,
    Downloading,
    Importing,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct GalleryImportProgress {
    pub status: GalleryImportStatus,
    pub gallery: u32,
    pub galleries: u32,
    pub pages: u32,
    pub total_pages: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct GalleryImportResult {
    pub project: String,
    pub pages: u32,
    pub manifest: String,
}

#[derive(Debug, Serialize)]
struct GalleryManifest {
    name: String,
    created_at: String,
    pages: usize,
    galleries: Vec<ManifestGallery>,
}

impl GalleryManifest {
    fn new(name: String, created_at: String, galleries: Vec<ManifestGallery>) -> Self {
        let pages = galleries.iter().map(|gallery| gallery.files.len()).sum();
        Self {
            name,
            created_at,
            pages,
            galleries,
        }
    }
}

#[derive(Debug, Serialize)]
struct ManifestGallery {
    index: usize,
    url: String,
    title: String,
    alternative_titles: Vec<String>,
    tags: Vec<String>,
    prefix: String,
    first_page: usize,
    last_page: usize,
    files: Vec<String>,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn import_galleries(
    request: GalleryImportRequest,
    on_event: Channel<GalleryImportProgress>,
    handle: AppHandle<Cef>,
) -> std::result::Result<GalleryImportResult, Error> {
    let urls = sources::normalize_urls(request.urls)?;
    let supplied_cookies = request
        .exhentai_cookies
        .and_then(|value| nonempty(&value).map(str::to_owned));
    let cookies = match supplied_cookies.clone() {
        Some(cookies) => Some(cookies),
        None => koharu_secrets::get(EXHENTAI_COOKIES_KEY)?
            .map(|cookies| cookies.expose_secret().to_owned()),
    };
    let sources = Sources::new(cookies)?;
    if let Some(cookies) = supplied_cookies {
        koharu_secrets::set(EXHENTAI_COOKIES_KEY, &cookies.into())?;
    }

    let prepared = prepare_import(&handle, &sources, urls, request.name, &on_event).await?;
    let pages = decode_import_paths(prepared.paths.clone()).await?;
    ensure_page_import_allowed(&handle)?;

    let library = handle.state::<ProjectLibrary>().inner().clone();
    let opened = library.create(&prepared.project).await?;
    replace_project(&handle, opened).await?;
    let imported = import_decoded_pages(&handle, pages).await?;
    if imported != prepared.manifest.pages {
        return Err(anyhow::anyhow!("gallery import decoded a different number of pages").into());
    }

    Ok(GalleryImportResult {
        project: prepared.project,
        pages: u32::try_from(imported).context("imported page count exceeds the UI limit")?,
        manifest: prepared.manifest_path.to_string_lossy().into_owned(),
    })
}

struct PreparedImport {
    project: String,
    paths: Vec<PathBuf>,
    manifest: GalleryManifest,
    manifest_path: PathBuf,
}

async fn prepare_import(
    handle: &AppHandle<Cef>,
    sources: &Sources,
    urls: Vec<String>,
    requested_name: Option<String>,
    on_event: &Channel<GalleryImportProgress>,
) -> Result<PreparedImport> {
    let library = handle.state::<ProjectLibrary>().inner().clone();
    let job = download_job()?;
    let directory = library.root().join("gallery-downloads").join(&job);
    tokio::fs::create_dir_all(&directory)
        .await
        .with_context(|| format!("failed to create {}", directory.display()))?;

    let galleries_total = progress_count(urls.len())?;
    let mut paths = Vec::new();
    let mut galleries = Vec::with_capacity(urls.len());
    let mut total_bytes = 0_u64;

    for (index, url) in urls.into_iter().enumerate() {
        let gallery_index = index + 1;
        publish(
            on_event,
            GalleryImportStatus::Inspecting,
            gallery_index,
            galleries_total,
            paths.len(),
            None,
        )?;
        let inspection = sources
            .inspect(&url, MAX_PAGES - paths.len())
            .await
            .with_context(|| format!("failed to inspect gallery {gallery_index}"))?;
        let total_pages = paths.len() + inspection.page_urls.len();
        if total_pages > MAX_PAGES {
            bail!("a gallery project is limited to {MAX_PAGES} pages");
        }
        publish(
            on_event,
            GalleryImportStatus::Downloading,
            gallery_index,
            galleries_total,
            paths.len(),
            Some(total_pages),
        )?;

        let prefix = format!("{gallery_index:02}-{}", slug(&inspection.title));
        let first_page = paths.len() + 1;
        let mut files = Vec::with_capacity(inspection.page_urls.len());
        for (page_index, page_url) in inspection.page_urls.iter().enumerate() {
            let page = sources.download(page_url).await.with_context(|| {
                format!(
                    "failed to download gallery {gallery_index} page {}",
                    page_index + 1
                )
            })?;
            total_bytes = total_bytes.saturating_add(page.bytes.len() as u64);
            if total_bytes > MAX_TOTAL_BYTES {
                bail!("gallery downloads exceed the 512 MiB total size limit");
            }
            let filename = page_filename(&prefix, page_index + 1, page.extension);
            let path = directory.join(&filename);
            tokio::fs::write(&path, page.bytes)
                .await
                .with_context(|| format!("failed to store {}", path.display()))?;
            paths.push(path);
            files.push(filename);
            publish(
                on_event,
                GalleryImportStatus::Downloading,
                gallery_index,
                galleries_total,
                paths.len(),
                Some(total_pages),
            )?;
        }
        if files.is_empty() {
            bail!("gallery {gallery_index} returned no pages");
        }
        galleries.push(manifest_gallery(
            gallery_index,
            url,
            inspection,
            prefix,
            first_page,
            paths.len(),
            files,
        ));
    }

    let name = import_name(requested_name, &galleries);
    let manifest = GalleryManifest::new(name, job.clone(), galleries);
    let manifest_path = directory.join("manifest.json");
    let encoded =
        serde_json::to_vec_pretty(&manifest).context("failed to encode gallery manifest")?;
    tokio::fs::write(&manifest_path, encoded)
        .await
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    publish(
        on_event,
        GalleryImportStatus::Importing,
        galleries_total as usize,
        galleries_total,
        paths.len(),
        Some(paths.len()),
    )?;

    let project = unique_project_name(&manifest.name, &job);
    Ok(PreparedImport {
        project,
        paths,
        manifest,
        manifest_path,
    })
}

fn manifest_gallery(
    index: usize,
    url: String,
    inspection: Inspection,
    prefix: String,
    first_page: usize,
    last_page: usize,
    files: Vec<String>,
) -> ManifestGallery {
    ManifestGallery {
        index,
        url,
        title: inspection.title,
        alternative_titles: inspection.alternative_titles,
        tags: inspection.tags,
        prefix,
        first_page,
        last_page,
        files,
    }
}

fn import_name(requested: Option<String>, galleries: &[ManifestGallery]) -> String {
    if let Some(name) = requested.and_then(|value| nonempty(&value).map(str::to_owned)) {
        return name;
    }
    let mut name = galleries
        .first()
        .map(|gallery| gallery.title.clone())
        .unwrap_or_else(|| "Gallery import".to_owned());
    if galleries.len() > 1 {
        name.push_str(&format!(" (+{})", galleries.len() - 1));
    }
    name
}

fn unique_project_name(title: &str, job: &str) -> String {
    let safe = title
        .chars()
        .map(|character| {
            if character.is_control() || r#"<>:"/\|?*"#.contains(character) {
                '_'
            } else {
                character
            }
        })
        .collect::<String>();
    let safe = safe.trim_matches([' ', '.']);
    let safe = if safe.is_empty() { "gallery" } else { safe };
    let base = safe.chars().take(70).collect::<String>();
    format!("{base} [{job}] - {}", Uuid::new_v4())
}

fn download_job() -> Result<String> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs();
    let identifier = Uuid::new_v4().simple().to_string();
    Ok(format!("{seconds}-{}", &identifier[..8]))
}

fn page_filename(prefix: &str, number: usize, extension: &str) -> String {
    format!("{prefix}-{number:03}.{extension}")
}

fn slug(title: &str) -> String {
    let stripped = strip_title_tags(title);
    let normalized = stripped.nfkc().collect::<String>().to_lowercase();
    let mut slug = String::new();
    let mut separator = false;
    for character in normalized.chars() {
        if character.is_alphanumeric() || character == '_' {
            if separator && !slug.is_empty() {
                slug.push('-');
            }
            slug.push(character);
            separator = false;
        } else {
            separator = true;
        }
    }
    let slug = slug
        .chars()
        .take(40)
        .collect::<String>()
        .trim_matches(['-', '_'])
        .to_owned();
    if slug.is_empty() {
        "gallery".to_owned()
    } else {
        slug
    }
}

fn strip_title_tags(title: &str) -> String {
    let mut value = title.to_owned();
    loop {
        let mut removed = None;
        for (close, closing) in value.char_indices() {
            let Some(expected) = matching_open(closing) else {
                continue;
            };
            if let Some((open, _)) = value[..close]
                .char_indices()
                .rev()
                .find(|(_, character)| *character == expected)
            {
                removed = Some((open, close + closing.len_utf8()));
                break;
            }
        }
        let Some((start, end)) = removed else {
            break;
        };
        value.replace_range(start..end, " ");
    }
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn matching_open(closing: char) -> Option<char> {
    match closing {
        ']' => Some('['),
        ')' => Some('('),
        '}' => Some('{'),
        '】' => Some('【'),
        '）' => Some('（'),
        _ => None,
    }
}

fn nonempty(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn progress_count(value: usize) -> Result<u32> {
    u32::try_from(value).context("gallery progress exceeds the UI limit")
}

fn publish(
    channel: &Channel<GalleryImportProgress>,
    status: GalleryImportStatus,
    gallery: usize,
    galleries: u32,
    pages: usize,
    total_pages: Option<usize>,
) -> Result<()> {
    let _ = channel.send(GalleryImportProgress {
        status,
        gallery: progress_count(gallery)?,
        galleries,
        pages: progress_count(pages)?,
        total_pages: total_pages.map(progress_count).transpose()?,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::Path};

    use super::*;

    #[test]
    fn filenames_use_natural_gallery_and_page_order() {
        let prefix = format!(
            "{:02}-{}",
            1,
            slug("(C99) [Circle (Artist)] First Story [English]")
        );
        assert_eq!(prefix, "01-first-story");
        assert_eq!(page_filename(&prefix, 1, "png"), "01-first-story-001.png");
        assert_eq!(
            page_filename(&prefix, 12, "webp"),
            "01-first-story-012.webp"
        );
    }

    #[test]
    fn slug_normalizes_unicode_and_has_a_nonempty_fallback() {
        assert_eq!(slug("  Ｆｕｌｌｗｉｄｔｈ / Story  "), "fullwidth-story");
        assert_eq!(slug("[English]"), "gallery");
    }

    #[test]
    fn manifest_preserves_source_order_and_gallery_boundaries() {
        let galleries = vec![
            ManifestGallery {
                index: 1,
                url: "https://e-hentai.org/g/1/a/".into(),
                title: "First".into(),
                alternative_titles: vec!["First JP".into()],
                tags: vec!["artist:one".into()],
                prefix: "01-first".into(),
                first_page: 1,
                last_page: 2,
                files: vec!["01-first-001.png".into(), "01-first-002.png".into()],
            },
            ManifestGallery {
                index: 2,
                url: "https://nhentai.net/g/2/".into(),
                title: "Second".into(),
                alternative_titles: vec![],
                tags: vec![],
                prefix: "02-second".into(),
                first_page: 3,
                last_page: 3,
                files: vec!["02-second-001.jpg".into()],
            },
        ];
        let manifest = GalleryManifest::new("Combined".into(), "job".into(), galleries);
        let encoded = serde_json::to_value(&manifest).unwrap();
        assert_eq!(encoded["pages"], 3);
        assert_eq!(
            encoded["galleries"][0]["url"],
            "https://e-hentai.org/g/1/a/"
        );
        assert_eq!(encoded["galleries"][0]["first_page"], 1);
        assert_eq!(encoded["galleries"][0]["last_page"], 2);
        assert_eq!(encoded["galleries"][1]["url"], "https://nhentai.net/g/2/");
        assert_eq!(encoded["galleries"][1]["first_page"], 3);
        assert_eq!(encoded["galleries"][1]["last_page"], 3);
        assert_eq!(encoded["galleries"][1]["files"][0], "02-second-001.jpg");
    }

    #[test]
    fn duplicate_free_manifest_paths_stay_under_the_download_directory() {
        let directory = Path::new("downloads").join("job");
        let first = directory.join(page_filename("01-first", 1, "png"));
        let second = directory.join(page_filename("02-second", 1, "png"));
        let names = [&first, &second]
            .into_iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<HashSet<_>>();
        assert_eq!(names.len(), 2);
        assert!(first.starts_with(&directory));
        assert!(second.starts_with(&directory));
    }
}
