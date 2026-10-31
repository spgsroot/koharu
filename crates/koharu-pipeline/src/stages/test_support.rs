//! Scene fixtures shared by the inpainting, decensoring, and censor tests.

use std::{collections::BTreeMap, io::Cursor, sync::Arc};

use image::{DynamicImage, ImageFormat, RgbaImage};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, EntityId, EntityOrigin, Origin, Patch, RasterLayer,
    RasterLayerKind, Snapshot,
};

pub(super) fn encode(image: &DynamicImage) -> Arc<[u8]> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png).unwrap();
    Arc::<[u8]>::from(bytes.into_inner())
}

/// An 8x8 mask made of `(x, y, coverage)` pixels, like a stroke from the editor.
pub(super) fn stroke(pixels: &[(u32, u32, u8)]) -> image::GrayImage {
    let mut mask = image::GrayImage::new(8, 8);
    for &(x, y, coverage) in pixels {
        mask.put_pixel(x, y, image::Luma([coverage]));
    }
    mask
}

/// A [`stroke`] painted for `target`, as the editor sends it with a request.
pub(super) fn painted(
    target: crate::CleanupTarget,
    pixels: &[(u32, u32, u8)],
) -> (crate::CleanupTarget, image::GrayImage) {
    (target, stroke(pixels))
}

pub(super) struct Fixture {
    pub(super) snapshot: Snapshot,
    pub(super) page: EntityId,
}

impl Fixture {
    /// Builds a page carrying `assets` and, when given, a committed cleanup layer.
    pub(super) async fn new(
        assets: &[(&str, DynamicImage)],
        cleanup: Option<(Origin, DynamicImage)>,
    ) -> Self {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                for (role, image) in assets {
                    edit.set_asset(
                        id,
                        &AssetRole::new(*role)?,
                        AssetInput::new(
                            encode(image),
                            "image/png",
                            AssetMetadata {
                                width: Some(8),
                                height: Some(8),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                if let Some((origin, image)) = cleanup {
                    let layer = edit.add_entity(id, At::Start)?;
                    edit.set(
                        layer,
                        &RasterLayer {
                            origin,
                            name: "Cleanup".to_owned(),
                            kind: RasterLayerKind::Cleanup,
                        },
                    )?;
                    edit.set_asset(
                        layer,
                        &AssetRole::new("source")?,
                        AssetInput::new(
                            encode(&image),
                            "image/png",
                            AssetMetadata {
                                width: Some(8),
                                height: Some(8),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        Self {
            snapshot,
            page: page.unwrap(),
        }
    }

    /// The input of one page run: `painted` is the stroke the request carries,
    /// with the target it was painted for, and `region` the requested region.
    pub(super) fn input(
        &self,
        painted: Option<(crate::CleanupTarget, image::GrayImage)>,
        region: Option<crate::Bounds>,
    ) -> super::StageInput {
        let page = self.page;
        super::StageInput::new(
            self.snapshot.clone(),
            page,
            None,
            region,
            Arc::new(crate::ImageCache::default()),
            painted.map(|(target, mask)| crate::InpaintingMask {
                page,
                png: encode(&DynamicImage::ImageLuma8(mask)),
                target,
            }),
            None,
        )
    }

    /// The fixture once `patch` is committed on top of it.
    pub(super) fn after(&self, patch: &Patch) -> Self {
        Self {
            snapshot: self.snapshot.preview([patch]).unwrap(),
            page: self.page,
        }
    }

    /// Every cleanup layer of the page with its layer and entity authorship.
    pub(super) fn cleanup_layers(&self) -> Vec<(RasterLayer, EntityOrigin)> {
        self.snapshot
            .children(self.page)
            .unwrap()
            .filter_map(|entity| {
                let layer = self.snapshot.component::<RasterLayer>(entity).unwrap()?;
                (layer.kind == RasterLayerKind::Cleanup).then(|| {
                    let origin = self.snapshot.component::<EntityOrigin>(entity).unwrap();
                    (layer, origin.unwrap())
                })
            })
            .collect()
    }

    /// The page as a viewer sees it: committed cleanup over the source.
    pub(super) async fn visible(&self) -> RgbaImage {
        let input = self.input(None, None);
        let source = input
            .images
            .get(&input.scene, input.page, "source")
            .await
            .unwrap()
            .unwrap();
        let layer = super::cleanup::load(&input, &source).await.unwrap();
        super::cleanup::composite(&source, &layer).to_rgba8()
    }
}
