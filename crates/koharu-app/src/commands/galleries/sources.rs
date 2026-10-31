//! Gallery-specific parsing. Each source owns its metadata, page ordering and
//! naming rules; the HTTP layer only provides validated, bounded responses.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use anyhow::{Context as _, Result, bail};
use html5ever::{
    tendril::StrTendril,
    tokenizer::{
        BufferQueue, TagKind, Token, TokenSink, TokenSinkResult, Tokenizer, TokenizerOpts,
    },
};
use image::ImageFormat;
use reqwest::Url;
use serde_json::{Value, json};

use super::MAX_GALLERIES;
use super::http::{Http, content_type};

const EHENTAI_HOSTS: &[&str] = &["e-hentai.org", "www.e-hentai.org"];
const EXHENTAI_HOSTS: &[&str] = &["exhentai.org", "www.exhentai.org"];
const NHENTAI_HOSTS: &[&str] = &["nhentai.net", "www.nhentai.net"];
const COOKIE_NAMES: &[&str] = &["ipb_member_id", "ipb_pass_hash", "igneous"];
const EHENTAI_LIMIT_TEXT: &[&str] = &["exceeded your image viewing limit", "bandwidth exceeded"];

#[derive(Debug)]
pub(crate) struct Inspection {
    pub(crate) title: String,
    pub(crate) alternative_titles: Vec<String>,
    pub(crate) tags: Vec<String>,
    pub(crate) page_urls: Vec<Url>,
}

#[derive(Debug)]
pub(crate) struct Downloaded {
    pub(crate) bytes: Vec<u8>,
    pub(crate) extension: &'static str,
}

pub(crate) struct Sources {
    http: Http,
    exhentai_cookies: Vec<(String, String)>,
}

impl Sources {
    pub(crate) fn new(cookies: Option<String>) -> Result<Self> {
        Ok(Self {
            http: Http::new(),
            exhentai_cookies: cookies.map_or_else(|| Ok(Vec::new()), parse_cookies)?,
        })
    }

    pub(crate) async fn inspect(&self, url: &str, max_pages: usize) -> Result<Inspection> {
        let url = Url::parse(url).context("source URL is invalid")?;
        // Validation happens before any source-specific request is issued.
        super::http::public_addresses(&url).await?;
        let source = host_of(&url)?.to_owned();
        if EHENTAI_HOSTS.contains(&source.as_str()) || EXHENTAI_HOSTS.contains(&source.as_str()) {
            self.inspect_ehentai(url, &source, max_pages).await
        } else if NHENTAI_HOSTS.contains(&source.as_str()) {
            self.inspect_nhentai(url, max_pages).await
        } else {
            bail!(
                "unsupported gallery host {source}; use e-hentai, exhentai or nhentai gallery links"
            )
        }
    }

    pub(crate) async fn download(&self, url: &Url) -> Result<Downloaded> {
        let response = self.http.get(url.clone(), &[]).await?;
        response.success()?;
        if response.body.is_empty() {
            bail!("source returned an empty image");
        }
        let format =
            image::guess_format(&response.body).context("source page is not a supported image")?;
        let (extension, mime) = match format {
            ImageFormat::Png => ("png", "image/png"),
            ImageFormat::Jpeg => ("jpg", "image/jpeg"),
            ImageFormat::WebP => ("webp", "image/webp"),
            _ => bail!("source page format is not supported; use PNG, JPEG or WebP"),
        };
        let declared = content_type(&response.headers);
        if !declared.is_empty() && declared != mime && declared != "application/octet-stream" {
            bail!("source page content type {declared} does not match its image data");
        }
        Ok(Downloaded {
            bytes: response.body,
            extension,
        })
    }

    async fn inspect_ehentai(&self, url: Url, host: &str, max_pages: usize) -> Result<Inspection> {
        let (gallery_id, token) = gallery_path(&url, "e-hentai")?;
        if EXHENTAI_HOSTS.contains(&host) && self.exhentai_cookies.len() != COOKIE_NAMES.len() {
            bail!(
                "exhentai requires ipb_member_id, ipb_pass_hash and igneous cookies in the import dialog"
            );
        }
        let api = Url::parse("https://api.e-hentai.org/api.php").unwrap();
        let response = self
            .http
            .post_json(
                api,
                json!({"method": "gdata", "gidlist": [[gallery_id, token]], "namespace": 1}),
                &self.site_headers("api.e-hentai.org"),
            )
            .await
            .context("e-hentai metadata request failed")?;
        response.success()?;
        let payload = response.json()?;
        if payload.get("error").is_some() {
            bail!("e-hentai metadata API rejected the gallery");
        }
        let metadata = payload
            .get("gmetadata")
            .and_then(Value::as_array)
            .and_then(|rows| if rows.len() == 1 { rows.first() } else { None })
            .context("e-hentai metadata API returned no gallery metadata")?;
        if metadata.get("error").is_some() {
            bail!("e-hentai gallery is unavailable or requires different access");
        }
        let page_count = metadata
            .get("filecount")
            .and_then(Value::as_str)
            .and_then(|value| value.parse::<usize>().ok())
            .or_else(|| {
                metadata
                    .get("filecount")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
            })
            .context("e-hentai gallery has an invalid page count")?;
        if page_count == 0 {
            bail!("e-hentai gallery contains no pages");
        }
        if page_count > max_pages {
            bail!("e-hentai gallery exceeds the remaining page limit");
        }
        let title = text(metadata, "title").or_else(|| text(metadata, "title_jpn"));
        let japanese = text(metadata, "title_jpn");
        let mut alternative_titles = Vec::new();
        for candidate in [japanese.clone(), title.clone()].into_iter().flatten() {
            if Some(&candidate) != title.as_ref() && !alternative_titles.contains(&candidate) {
                alternative_titles.push(candidate);
            }
        }
        let mut tags = Vec::new();
        if let Some(category) = text(metadata, "category") {
            tags.push(format!("category:{category}"));
        }
        if let Some(rows) = metadata.get("tags").and_then(Value::as_array) {
            tags.extend(rows.iter().filter_map(Value::as_str).map(str::to_owned));
        }

        let origin =
            Url::parse(&format!("{}://{}", url.scheme(), url.host_str().unwrap())).unwrap();
        let mut showpages = BTreeMap::new();
        for listing_index in 0..page_count {
            if showpages.len() >= page_count {
                break;
            }
            let listing_url = origin
                .join(&format!("/g/{gallery_id}/{token}/?p={listing_index}"))
                .unwrap();
            let response = self.http.get(listing_url, &self.site_headers(host)).await?;
            response.success()?;
            let html = response.text();
            ehentai_error_page(&html, None)?;
            let before = showpages.len();
            for tag in parse_tags(&html) {
                if tag.name != "a" {
                    continue;
                }
                let Some(href) = tag.attribute("href") else {
                    continue;
                };
                let Ok(link) = response.url.join(href) else {
                    continue;
                };
                // Ordinary links (navigation, forums, other galleries) are
                // ignored; only links shaped like this gallery's showpages count.
                let marker = format!("/{gallery_id}-");
                let Some((prefix, number)) = link.path().rsplit_once(&marker) else {
                    continue;
                };
                if !prefix.starts_with("/s/") {
                    continue;
                }
                let Ok(number) = number.trim_end_matches('/').parse::<usize>() else {
                    continue;
                };
                let link_host = host_of(&link)?;
                if !EHENTAI_HOSTS.contains(&link_host) && !EXHENTAI_HOSTS.contains(&link_host) {
                    bail!("e-hentai gallery returned a showpage on an unexpected host");
                }
                if number == 0 || number > page_count {
                    bail!("e-hentai gallery returned an out-of-range page");
                }
                showpages.insert(number, link);
            }
            if showpages.len() == before {
                bail!("e-hentai gallery listing returned no new pages");
            }
        }
        if showpages.len() != page_count {
            bail!("e-hentai gallery listing is incomplete");
        }

        let mut page_urls = Vec::with_capacity(page_count);
        for (number, showpage_url) in showpages {
            let response = self
                .http
                .get(showpage_url, &self.site_headers(host))
                .await?;
            response.success()?;
            let html = response.text();
            let tags = parse_tags(&html);
            let image = tags
                .iter()
                .find(|tag| tag.name == "img" && tag.attribute("id") == Some("img"))
                .and_then(|tag| tag.attribute("src"))
                .map(|src| response.url.join(src))
                .transpose()
                .context("e-hentai returned an invalid image URL")?
                .with_context(|| format!("e-hentai showpage {number} returned no full image"))?;
            ehentai_error_page(&html, Some(&image))?;
            page_urls.push(image);
        }
        Ok(Inspection {
            title: title.unwrap_or_else(|| format!("e-hentai gallery {gallery_id}")),
            alternative_titles,
            tags,
            page_urls,
        })
    }

    async fn inspect_nhentai(&self, url: Url, max_pages: usize) -> Result<Inspection> {
        let (gallery_id, _) = gallery_path(&url, "nhentai")?;
        let gallery = self
            .read_json(
                Url::parse(&format!(
                    "https://nhentai.net/api/v2/galleries/{gallery_id}"
                ))
                .unwrap(),
                "nhentai gallery",
            )
            .await?;
        let cdn = self
            .read_json(
                Url::parse("https://nhentai.net/api/v2/cdn").unwrap(),
                "nhentai CDN",
            )
            .await?;
        let raw_pages = gallery
            .get("pages")
            .and_then(Value::as_array)
            .filter(|pages| !pages.is_empty())
            .context("nhentai gallery API returned no page paths")?;
        let mut pages = Vec::with_capacity(raw_pages.len());
        for raw in raw_pages {
            let number =
                raw.get("number")
                    .and_then(Value::as_u64)
                    .context("nhentai returned an invalid page number")? as usize;
            let path = raw
                .get("path")
                .and_then(Value::as_str)
                .context("nhentai returned an invalid page path")?;
            if number == 0 || pages.iter().any(|(seen, _)| *seen == number) {
                bail!("nhentai returned a duplicate or zero page number");
            }
            validate_nhentai_path(path)?;
            pages.push((number, path.to_owned()));
        }
        pages.sort_by_key(|(number, _)| *number);
        if let Some(expected) = gallery.get("num_pages").and_then(Value::as_u64)
            && expected as usize != pages.len()
        {
            bail!("nhentai gallery page count does not match its metadata");
        }
        if pages.len() > max_pages {
            bail!("nhentai gallery exceeds the remaining page limit");
        }
        let servers = cdn
            .get("image_servers")
            .and_then(Value::as_array)
            .context("nhentai CDN API returned no image servers")?;
        let mut page_urls = None;
        for server in servers.iter().filter_map(Value::as_str) {
            let Ok(base) = Url::parse(&format!("{}/", server.trim_end_matches('/'))) else {
                continue;
            };
            if base.scheme() != "https" || !base.username().is_empty() || base.password().is_some()
            {
                continue;
            }
            let candidates = pages
                .iter()
                .map(|(_, path)| base.join(path))
                .collect::<std::result::Result<Vec<_>, _>>();
            if let Ok(candidates) = candidates {
                page_urls = Some(candidates);
                break;
            }
        }
        let page_urls =
            page_urls.context("nhentai image servers are not allowed by source policy")?;

        let raw_title = gallery.get("title");
        let title = raw_title
            .and_then(|titles| {
                text(titles, "pretty")
                    .or_else(|| text(titles, "english"))
                    .or_else(|| text(titles, "japanese"))
            })
            .unwrap_or_else(|| format!("nhentai gallery {gallery_id}"));
        let mut alternative_titles = Vec::new();
        if let Some(titles) = raw_title {
            for candidate in [text(titles, "english"), text(titles, "japanese")]
                .into_iter()
                .flatten()
            {
                if candidate != title && !alternative_titles.contains(&candidate) {
                    alternative_titles.push(candidate);
                }
            }
        }
        let tags = gallery
            .get("tags")
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .filter_map(|tag| Some((text(tag, "type")?, text(tag, "name")?)))
                    .map(|(kind, name)| format!("{kind}:{name}"))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Inspection {
            title,
            alternative_titles,
            tags,
            page_urls,
        })
    }

    async fn read_json(&self, url: Url, label: &str) -> Result<Value> {
        let response = self.http.get(url, &[]).await?;
        response.success()?;
        response
            .json()
            .with_context(|| format!("{label} returned an invalid response"))
    }

    fn site_headers(&self, _host: &str) -> Vec<(String, String)> {
        if self.exhentai_cookies.is_empty() {
            return Vec::new();
        }
        let value = self
            .exhentai_cookies
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join("; ");
        vec![("Cookie".to_owned(), value)]
    }
}

pub(crate) fn normalize_urls(urls: Vec<String>) -> Result<Vec<String>> {
    let mut cleaned = Vec::new();
    for raw in urls {
        for line in raw.lines() {
            let value = line.trim();
            if !value.is_empty() && !cleaned.iter().any(|seen| seen == value) {
                cleaned.push(value.to_owned());
            }
        }
    }
    if cleaned.is_empty() {
        bail!("add at least one gallery link");
    }
    if cleaned.len() > MAX_GALLERIES {
        bail!("no more than {MAX_GALLERIES} gallery links per import");
    }
    for value in &cleaned {
        let url = Url::parse(value).with_context(|| format!("invalid gallery URL: {value}"))?;
        if url.scheme() != "https" {
            bail!("gallery links must use HTTPS: {value}");
        }
        if !url.username().is_empty() || url.password().is_some() {
            bail!("gallery links must not contain credentials");
        }
        let host = host_of(&url)?;
        if EHENTAI_HOSTS.contains(&host) || EXHENTAI_HOSTS.contains(&host) {
            gallery_path(&url, "e-hentai")?;
        } else if NHENTAI_HOSTS.contains(&host) {
            gallery_path(&url, "nhentai")?;
        } else {
            bail!("unsupported gallery host {host}");
        }
    }
    Ok(cleaned)
}

fn parse_cookies(raw: String) -> Result<Vec<(String, String)>> {
    let mut values = Vec::new();
    for part in raw.split(';') {
        let Some((name, value)) = part.trim().split_once('=') else {
            bail!(
                "exhentai cookies must be a cookie header such as ipb_member_id=…; ipb_pass_hash=…; igneous=…"
            );
        };
        let name = name.trim();
        let value = value.trim();
        if !COOKIE_NAMES.contains(&name) || value.is_empty() || value.contains(['\r', '\n', ';']) {
            bail!("exhentai cookie header contains an unsupported or empty value");
        }
        if !values.iter().any(|(seen, _)| seen == name) {
            values.push((name.to_owned(), value.to_owned()));
        }
    }
    if values.len() != COOKIE_NAMES.len() {
        bail!("exhentai requires ipb_member_id, ipb_pass_hash and igneous cookies");
    }
    Ok(values)
}

fn gallery_path(url: &Url, source: &str) -> Result<(u64, String)> {
    let segments = url
        .path_segments()
        .map(|segments| {
            segments
                .filter(|segment| !segment.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    match (source, segments.as_slice()) {
        ("e-hentai", ["g", id, token]) => {
            let id = id.parse().context("e-hentai gallery id is invalid")?;
            if token.is_empty() || !token.chars().all(|character| character.is_ascii_hexdigit()) {
                bail!("e-hentai gallery token is invalid");
            }
            Ok((id, (*token).to_owned()))
        }
        ("nhentai", ["g", id]) => Ok((
            id.parse().context("nhentai gallery id is invalid")?,
            String::new(),
        )),
        _ => bail!("{source} source must be a gallery URL such as /g/123/token/"),
    }
}

fn validate_nhentai_path(path: &str) -> Result<()> {
    if path.starts_with('/')
        || path.contains(['\\', '?', '#'])
        || path.split('/').any(|part| part == ".." || part.is_empty())
    {
        bail!("nhentai returned an unsafe page path");
    }
    Ok(())
}

fn ehentai_error_page(html: &str, image: Option<&Url>) -> Result<()> {
    if html.trim().is_empty() {
        bail!("e-hentai returned an empty response; check gallery access and exhentai cookies");
    }
    let lower = html.to_lowercase();
    if EHENTAI_LIMIT_TEXT
        .iter()
        .any(|marker| lower.contains(marker))
        || image.is_some_and(|image| image.path().ends_with("/509.gif"))
    {
        bail!("e-hentai image/bandwidth limit reached (509); retry after the limit resets");
    }
    Ok(())
}

fn host_of(url: &Url) -> Result<&str> {
    url.host_str()
        .map(|host| host.trim_end_matches('.'))
        .filter(|host| !host.is_empty())
        .context("source URL has no host")
}

fn text(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

#[derive(Debug)]
struct Tag {
    name: String,
    attributes: Vec<(String, String)>,
}

impl Tag {
    fn attribute(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

#[derive(Clone, Default)]
struct TagSink {
    tags: Rc<RefCell<Vec<Tag>>>,
}

impl TokenSink for TagSink {
    type Handle = ();

    fn process_token(&self, token: Token, _line_number: u64) -> TokenSinkResult<Self::Handle> {
        if let Token::TagToken(tag) = token
            && tag.kind == TagKind::StartTag
        {
            self.tags.borrow_mut().push(Tag {
                name: tag.name.to_string(),
                attributes: tag
                    .attrs
                    .iter()
                    .map(|attribute| {
                        (
                            attribute.name.local.to_string(),
                            attribute.value.to_string(),
                        )
                    })
                    .collect(),
            });
        }
        TokenSinkResult::Continue
    }
}

fn parse_tags(html: &str) -> Vec<Tag> {
    let sink = TagSink::default();
    let tags = sink.tags.clone();
    let tokenizer = Tokenizer::new(sink, TokenizerOpts::default());
    let input = BufferQueue::default();
    input.push_back(StrTendril::from(html));
    let _ = tokenizer.feed(&input);
    tokenizer.end();
    tags.take()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gallery_links_are_deduplicated_in_order_and_validated() {
        let urls = normalize_urls(vec![
            " https://nhentai.net/g/123/ \nhttps://e-hentai.org/g/4/abcd/".into(),
            "https://nhentai.net/g/123/".into(),
        ])
        .unwrap();
        assert_eq!(
            urls,
            [
                "https://nhentai.net/g/123/",
                "https://e-hentai.org/g/4/abcd/"
            ]
        );
        for invalid in [
            "https://nhentai.net/",
            "https://example.com/g/123/",
            "http://nhentai.net/g/123/",
            "https://user:pw@nhentai.net/g/123/",
        ] {
            assert!(
                normalize_urls(vec![invalid.into()]).is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn tokenizer_extracts_showpages_and_the_full_image() {
        let listing = parse_tags(
            r#"<a href="https://e-hentai.org/s/abc/618395-2">page</a><img id="img" src="https://node.hath.network/h/2.jpg">"#,
        );
        assert_eq!(listing[0].name, "a");
        assert!(listing[0].attribute("href").unwrap().contains("618395-2"));
        assert_eq!(
            listing[1].attribute("src"),
            Some("https://node.hath.network/h/2.jpg")
        );
    }

    #[test]
    fn nhentai_paths_cannot_escape_the_image_server() {
        assert!(validate_nhentai_path("galleries/123/1.jpg").is_ok());
        for path in [
            "/etc/passwd",
            "../1.jpg",
            "galleries//1.jpg",
            "galleries/1.jpg?x=1",
        ] {
            assert!(validate_nhentai_path(path).is_err(), "accepted {path}");
        }
    }

    #[test]
    fn cookie_header_requires_exactly_the_three_exhentai_secrets() {
        let parsed =
            parse_cookies("ipb_member_id=42; ipb_pass_hash=hash; igneous=ig".into()).unwrap();
        assert_eq!(parsed.len(), 3);
        assert!(parse_cookies("ipb_member_id=42".into()).is_err());
        assert!(
            parse_cookies("ipb_member_id=42; ipb_pass_hash=hash; igneous=ig; other=x".into())
                .is_err()
        );
    }
}
