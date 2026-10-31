use std::{io::Cursor, net::IpAddr, time::Duration};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use image::{DynamicImage, GenericImageView as _, GrayImage, ImageFormat, Rgba, RgbaImage};
use reqwest::{Client, Url, multipart};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::{Instant, sleep};
use uuid::Uuid;

use super::models::DiffusionCreamPyConfig;

const WORKFLOW: &str = include_str!("../../workflows/diffusioncreampy-sdxl-api.json");
const OUTPUT_NODE: &str = "19";
const REQUIRED_NODES: &[&str] = &[
    "ACN_AdvancedControlNetApply_v2",
    "ACN_ControlNet++InputNode",
    "ACN_ControlNet++LoaderAdvanced",
    "CropByMask",
    "INPAINT_ColorMatch",
    "INPAINT_ExpandMask",
    "ImageAndMaskResizeNode",
    "ImageCropMerge",
    "InpaintPreprocessor",
];

#[derive(Clone)]
pub(super) struct ComfyUi {
    endpoint: Url,
    http: Client,
}

impl ComfyUi {
    pub(super) async fn connect(config: &DiffusionCreamPyConfig) -> Result<Self> {
        let endpoint = endpoint_url(&config.endpoint)?;
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(30))
            .user_agent(concat!("koharu/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("failed to create ComfyUI HTTP client")?;
        let object_info = success(
            http.get(endpoint.join("object_info")?)
                .send()
                .await
                .context("failed to reach local ComfyUI")?,
        )
        .await?
        .json::<Value>()
        .await
        .context("ComfyUI returned invalid object_info JSON")?;
        let missing = REQUIRED_NODES
            .iter()
            .filter(|name| object_info.get(**name).is_none())
            .copied()
            .collect::<Vec<_>>();
        ensure!(
            missing.is_empty(),
            "ComfyUI is missing required DiffusionCreamPy nodes: {}",
            missing.join(", ")
        );
        ensure_input_option(
            &object_info,
            "CheckpointLoaderSimple",
            "ckpt_name",
            &config.checkpoint,
            "checkpoint",
        )?;
        ensure_input_option(
            &object_info,
            "ACN_ControlNet++LoaderAdvanced",
            "name",
            &config.controlnet,
            "ControlNet",
        )?;
        Ok(Self { endpoint, http })
    }

    pub(super) async fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &DiffusionCreamPyConfig,
    ) -> Result<DynamicImage> {
        ensure!(
            image.dimensions() == mask.dimensions(),
            "ComfyUI image and mask dimensions differ"
        );
        let request_id = Uuid::new_v4().simple().to_string();
        let input_name = self
            .upload_masked_png(image, mask, &format!("koharu-{request_id}.png"))
            .await?;
        let mut workflow: Value = serde_json::from_str(WORKFLOW)
            .context("embedded DiffusionCreamPy workflow is invalid JSON")?;
        set_input(&mut workflow, "1", "image", json!(input_name))?;
        set_input(&mut workflow, "2", "image", json!(input_name))?;
        set_input(&mut workflow, "3", "padding", json!(config.mask_padding))?;
        set_input(&mut workflow, "5", "ckpt_name", json!(config.checkpoint))?;
        set_input(&mut workflow, "6", "text", json!(config.positive_prompt))?;
        set_input(&mut workflow, "7", "text", json!(config.negative_prompt))?;
        set_input(&mut workflow, "10", "name", json!(config.controlnet))?;
        set_input(&mut workflow, "13", "seed", json!(config.seed))?;
        set_input(
            &mut workflow,
            "13",
            "steps",
            json!(config.num_inference_steps),
        )?;
        set_input(&mut workflow, "13", "cfg", json!(config.guidance_scale))?;
        set_input(&mut workflow, "13", "denoise", json!(config.denoise))?;
        set_input(&mut workflow, "15", "grow", json!(config.mask_grow))?;
        set_input(&mut workflow, "15", "blur", json!(config.mask_blur))?;
        set_input(
            &mut workflow,
            "16",
            "strength",
            json!(config.color_match_strength),
        )?;
        set_input(
            &mut workflow,
            OUTPUT_NODE,
            "filename_prefix",
            json!(format!("koharu/{request_id}")),
        )?;

        let client_id = Uuid::new_v4().to_string();
        let response = success(
            self.http
                .post(self.endpoint.join("prompt")?)
                .json(&json!({ "prompt": workflow, "client_id": client_id }))
                .send()
                .await?,
        )
        .await?
        .json::<PromptResponse>()
        .await
        .context("ComfyUI returned invalid prompt response")?;
        self.wait_for_output(&response.prompt_id, config.timeout_seconds)
            .await
    }

    async fn upload_masked_png(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        filename: &str,
    ) -> Result<String> {
        let rgb = image.to_rgb8();
        let mut rgba = RgbaImage::new(rgb.width(), rgb.height());
        for (x, y, pixel) in rgba.enumerate_pixels_mut() {
            let source = rgb.get_pixel(x, y).0;
            *pixel = Rgba([
                source[0],
                source[1],
                source[2],
                255 - mask.get_pixel(x, y)[0],
            ]);
        }
        let mut png = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(rgba)
            .write_to(&mut png, ImageFormat::Png)
            .context("failed to encode ComfyUI input PNG")?;
        let part = multipart::Part::bytes(png.into_inner())
            .file_name(filename.to_owned())
            .mime_str("image/png")?;
        let upload = success(
            self.http
                .post(self.endpoint.join("upload/image")?)
                .multipart(
                    multipart::Form::new()
                        .text("type", "input")
                        .text("subfolder", "koharu")
                        .text("overwrite", "false")
                        .part("image", part),
                )
                .send()
                .await?,
        )
        .await?
        .json::<UploadedImage>()
        .await
        .context("ComfyUI returned invalid upload response")?;
        Ok(if upload.subfolder.is_empty() {
            upload.name
        } else {
            format!("{}/{}", upload.subfolder, upload.name)
        })
    }

    async fn wait_for_output(&self, prompt_id: &str, timeout_seconds: u32) -> Result<DynamicImage> {
        let deadline = Instant::now() + Duration::from_secs(u64::from(timeout_seconds));
        loop {
            let history = success(
                self.http
                    .get(self.endpoint.join(&format!("history/{prompt_id}"))?)
                    .send()
                    .await?,
            )
            .await?
            .json::<Value>()
            .await
            .context("ComfyUI returned invalid history JSON")?;
            if let Some(entry) = history.get(prompt_id) {
                if let Some(message) = execution_error(entry) {
                    bail!("ComfyUI workflow failed: {message}");
                }
                if let Some(image) = output_image(entry) {
                    return self.download_output(image).await;
                }
            }
            if Instant::now() >= deadline {
                bail!("ComfyUI workflow timed out after {timeout_seconds} seconds");
            }
            sleep(Duration::from_millis(250)).await;
        }
    }

    async fn download_output(&self, image: OutputImage) -> Result<DynamicImage> {
        let response = success(
            self.http
                .get(self.endpoint.join("view")?)
                .query(&[
                    ("filename", image.filename.as_str()),
                    ("subfolder", image.subfolder.as_str()),
                    ("type", image.kind.as_str()),
                ])
                .send()
                .await?,
        )
        .await?;
        let bytes = response
            .bytes()
            .await
            .context("failed to read ComfyUI output")?;
        image::load_from_memory(&bytes).context("ComfyUI returned an invalid output image")
    }
}

fn endpoint_url(endpoint: &str) -> Result<Url> {
    let mut url = Url::parse(endpoint).context("invalid ComfyUI endpoint")?;
    ensure!(url.scheme() == "http", "ComfyUI endpoint must use http");
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "ComfyUI endpoint must not contain credentials"
    );
    ensure!(
        url.query().is_none() && url.fragment().is_none(),
        "invalid ComfyUI endpoint"
    );
    let loopback = match url.host_str() {
        Some("localhost") => true,
        Some(host) => host
            .strip_prefix('[')
            .and_then(|host| host.strip_suffix(']'))
            .unwrap_or(host)
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback()),
        None => false,
    };
    ensure!(loopback, "ComfyUI endpoint must be loopback-only");
    if url.path().is_empty() {
        url.set_path("/");
    } else {
        ensure!(
            url.path() == "/",
            "ComfyUI endpoint must not contain a path"
        );
    }
    Ok(url)
}

fn set_input(workflow: &mut Value, node: &str, name: &str, value: Value) -> Result<()> {
    let inputs = workflow
        .get_mut(node)
        .and_then(|node| node.get_mut("inputs"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow!("embedded workflow node {node} has no inputs"))?;
    ensure!(
        inputs.insert(name.to_owned(), value).is_some(),
        "embedded workflow node {node} has no {name} input"
    );
    Ok(())
}

fn ensure_input_option(
    object_info: &Value,
    node: &str,
    input: &str,
    expected: &str,
    kind: &str,
) -> Result<()> {
    let schema = object_info
        .get(node)
        .and_then(|node| node.get("input"))
        .and_then(|input| input.get("required"))
        .and_then(|required| required.get(input))
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("ComfyUI node {node} has no {input} input"))?;
    let options = schema
        .first()
        .and_then(Value::as_array)
        .or_else(|| {
            schema
                .get(1)
                .and_then(|metadata| metadata.get("options"))
                .and_then(Value::as_array)
        })
        .ok_or_else(|| anyhow!("ComfyUI node {node} exposes no {input} options"))?;
    ensure!(
        options
            .iter()
            .any(|option| option.as_str() == Some(expected)),
        "ComfyUI {kind} `{expected}` is not installed"
    );
    Ok(())
}
async fn success(response: reqwest::Response) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    bail!("ComfyUI returned HTTP {status}: {body}")
}

fn execution_error(entry: &Value) -> Option<String> {
    let status = entry.get("status")?;
    if status.get("status_str")?.as_str()? != "error" {
        return None;
    }
    status
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.last())
        .map(ToString::to_string)
        .or_else(|| Some("unknown execution error".to_owned()))
}

fn output_image(entry: &Value) -> Option<OutputImage> {
    let image = entry
        .get("outputs")?
        .get(OUTPUT_NODE)?
        .get("images")?
        .as_array()?
        .first()?;
    serde_json::from_value(image.clone()).ok()
}

#[derive(Deserialize)]
struct PromptResponse {
    prompt_id: String,
}

#[derive(Deserialize)]
struct UploadedImage {
    name: String,
    #[serde(default)]
    subfolder: String,
}

#[derive(Deserialize)]
struct OutputImage {
    filename: String,
    #[serde(default)]
    subfolder: String,
    #[serde(rename = "type")]
    kind: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_is_limited_to_loopback_http() {
        assert!(endpoint_url("http://127.0.0.1:8188").is_ok());
        assert!(endpoint_url("http://[::1]:8188").is_ok());
        assert!(endpoint_url("https://127.0.0.1:8188").is_err());
        assert!(endpoint_url("http://example.com:8188").is_err());
        assert!(endpoint_url("http://127.0.0.1:8188/api").is_err());
    }
}
