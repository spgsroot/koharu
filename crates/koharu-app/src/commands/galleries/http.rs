//! Bounded HTTP transport for untrusted gallery sources.
//!
//! Every hop is validated and DNS-pinned so a redirect or a rebinding answer
//! cannot turn a public gallery link into a request to a private address.

use std::{
    net::{IpAddr, SocketAddr},
    time::Duration,
};

use anyhow::{Context as _, Result, bail};
use futures::StreamExt as _;
use reqwest::{
    Client, Response, StatusCode, Url,
    header::{self, HeaderMap, HeaderName, HeaderValue},
    redirect::Policy,
};
use serde_json::Value;

const MAX_REDIRECTS: usize = 5;
pub(crate) const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
const USER_AGENT: &str = "Koharu gallery importer/1.0";

#[derive(Clone, Debug)]
pub(crate) struct Http {
    timeout: Duration,
}

#[derive(Debug)]
pub(crate) struct HttpResponse {
    pub(crate) url: Url,
    pub(crate) status: StatusCode,
    pub(crate) headers: HeaderMap,
    pub(crate) body: Vec<u8>,
}

impl HttpResponse {
    pub(crate) fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    pub(crate) fn json(&self) -> Result<Value> {
        serde_json::from_slice(&self.body).context("source returned invalid JSON")
    }

    pub(crate) fn success(&self) -> Result<()> {
        if self.status.is_success() {
            Ok(())
        } else {
            bail!(
                "source request to {} returned HTTP {}",
                self.url.host_str().unwrap_or("source"),
                self.status
            )
        }
    }
}

impl Http {
    pub(crate) fn new() -> Self {
        Self {
            timeout: Duration::from_secs(45),
        }
    }

    pub(crate) async fn get(&self, url: Url, headers: &[(String, String)]) -> Result<HttpResponse> {
        self.send(url, Method::Get, headers, None).await
    }

    pub(crate) async fn post_json(
        &self,
        url: Url,
        body: Value,
        headers: &[(String, String)],
    ) -> Result<HttpResponse> {
        self.send(url, Method::Post, headers, Some(body)).await
    }

    async fn send(
        &self,
        url: Url,
        method: Method,
        headers: &[(String, String)],
        json: Option<Value>,
    ) -> Result<HttpResponse> {
        let mut current = url;
        let original_host = current.host_str().unwrap_or_default().to_owned();
        for _ in 0..=MAX_REDIRECTS {
            let addresses = public_addresses(&current).await?;
            let client = Client::builder()
                .user_agent(USER_AGENT)
                .timeout(self.timeout)
                .redirect(Policy::none())
                .no_proxy()
                .resolve_to_addrs(
                    current.host_str().context("source URL has no host")?,
                    &addresses,
                )
                .build()
                .context("failed to build the source HTTP client")?;
            let mut request = match method {
                Method::Get => client.get(current.clone()),
                Method::Post => client.post(current.clone()),
            };
            for (name, value) in headers {
                if name.eq_ignore_ascii_case("cookie")
                    && current.host_str().unwrap_or_default() != original_host
                {
                    continue;
                }
                request = request.header(
                    HeaderName::from_bytes(name.as_bytes())
                        .context("invalid source header name")?,
                    HeaderValue::from_str(value).context("invalid source header value")?,
                );
            }
            if let Some(body) = &json {
                request = request.json(body);
            }
            let response = request.send().await.context("source request failed")?;
            if response.status().is_redirection() {
                let Some(location) = response.headers().get(header::LOCATION) else {
                    bail!("source redirect did not include a location");
                };
                let location = location
                    .to_str()
                    .context("source redirect location is not valid text")?;
                current = current
                    .join(location)
                    .context("source redirect URL is invalid")?;
                continue;
            }
            return read_response(response).await;
        }
        bail!("source exceeded {MAX_REDIRECTS} redirects")
    }
}

#[derive(Clone, Copy)]
enum Method {
    Get,
    Post,
}

async fn read_response(response: Response) -> Result<HttpResponse> {
    let url = response.url().clone();
    let status = response.status();
    let headers = response.headers().clone();
    if let Some(length) = response.content_length()
        && length > MAX_BODY_BYTES as u64
    {
        bail!(
            "source response exceeds the {} MiB limit",
            MAX_BODY_BYTES / 1024 / 1024
        );
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("source response stream failed")?;
        if body.len() + chunk.len() > MAX_BODY_BYTES {
            bail!(
                "source response exceeds the {} MiB limit",
                MAX_BODY_BYTES / 1024 / 1024
            );
        }
        body.extend_from_slice(&chunk);
    }
    Ok(HttpResponse {
        url,
        status,
        headers,
        body,
    })
}

/// Resolves and pins only globally routable peers; credentials and non-HTTP(S)
/// schemes are rejected before any request is made. Ports are not restricted:
/// e-hentai serves images from H@H nodes on arbitrary public ports.
pub(crate) async fn public_addresses(url: &Url) -> Result<Vec<SocketAddr>> {
    if !matches!(url.scheme(), "https" | "http") {
        bail!("source URL must use HTTP(S)");
    }
    if url.scheme() == "http" && !cfg!(test) {
        bail!("source URL must use HTTPS");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("source URL must not contain credentials");
    }
    let host = url
        .host_str()
        .context("source URL has no host")?
        .trim_end_matches('.');
    if host.is_empty() {
        bail!("source URL has no host");
    }
    let port = url
        .port_or_known_default()
        .context("source URL has no port")?;
    let addresses = tokio::net::lookup_host((host, port))
        .await
        .with_context(|| format!("source hostname {host} could not be resolved"))?
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        bail!("source hostname {host} resolved to no addresses");
    }
    for address in &addresses {
        if !is_global(address.ip()) {
            bail!("source target {host} is not globally routable");
        }
    }
    Ok(addresses)
}

/// Stable-Rust equivalent of the unstable `IpAddr::is_global`: only public
/// unicast destinations are acceptable for an untrusted link.
fn is_global(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(ip.is_unspecified()
                || ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_multicast()
                || a == 0
                || (a == 100 && (64..=127).contains(&b))
                || (a == 192 && b == 0 && c == 0)
                || (a == 192 && b == 0 && c == 2)
                || (a == 198 && (b == 18 || b == 19))
                || (a == 198 && b == 51 && c == 100)
                || (a == 203 && b == 0 && c == 113)
                || a >= 240)
        }
        IpAddr::V6(ip) => {
            if let Some(mapped) = ip.to_ipv4_mapped() {
                return is_global(IpAddr::V4(mapped));
            }
            let segments = ip.segments();
            !(ip.is_unspecified()
                || ip.is_loopback()
                || ip.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || (segments[0] == 0x2001 && segments[1] == 0x0db8)
                || segments[0] == 0x0100 && segments[1..4] == [0, 0, 0])
        }
    }
}

pub(crate) fn content_type(headers: &HeaderMap) -> &str {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn private_and_credentialled_targets_are_rejected_before_dialing() {
        for raw in [
            "https://127.0.0.1/g/1/a/",
            "https://192.168.1.4/g/1/a/",
            "https://user:pw@example.com/g/1/a/",
            "file:///etc/passwd",
            "https://127.0.0.1:8443/g/1/a/",
        ] {
            let url = Url::parse(raw).unwrap();
            assert!(public_addresses(&url).await.is_err(), "accepted {raw}");
        }
    }
}
