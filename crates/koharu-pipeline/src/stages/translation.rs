use anyhow::Result;
use async_trait::async_trait;
use koharu_scene::{Authored, EntityId, LanguageTag, Origin, Snapshot, SourceText, Translation};
use koharu_translator::{
    ContentProfile, Language, TranslationContext, TranslationRequest, TranslationValidator,
    Translator,
};

use crate::{TranslationConfig, TranslationOptions};

use super::{StageInput, StageProcessor, finish, generation};

const PRODUCER: &str = "dev.koharu.pipeline.translation";

pub(super) struct Processor {
    config: TranslationConfig,
    validator: TranslationValidator,
    translator: Translator,
}

impl Processor {
    pub(super) fn new(config: TranslationConfig, translator: Translator) -> Result<Self> {
        let validator = TranslationValidator::new(&config.validators)?;
        Ok(Self {
            config,
            validator,
            translator,
        })
    }
}

fn resolved_options<'a>(
    config: &'a TranslationConfig,
    options: Option<&TranslationOptions>,
) -> (
    koharu_translator::Language,
    koharu_translator::Language,
    ContentProfile,
    Option<&'a str>,
) {
    let source_language = options
        .map(|options| options.source_language)
        .unwrap_or(config.source_language);
    let target_language = options
        .map(|options| options.target_language)
        .unwrap_or(config.target_language);
    let profile = options
        .map(|options| options.content_profile)
        .unwrap_or(config.content_profile);
    (
        source_language,
        target_language,
        profile,
        config.system_prompts.get(profile),
    )
}

fn translation_context(
    scene: &Snapshot,
    current_page: EntityId,
    target_language: Language,
) -> Result<Vec<TranslationContext>> {
    let mut context = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for page in scene.pages().take_while(|page| page.id() != current_page) {
        let Some(group) = page.text_group()? else {
            continue;
        };
        for layer in group.text_layers()? {
            let content = layer.content()?;
            if !seen.insert(content.id()) {
                continue;
            }
            let (Some(source), Some(translation)) = (content.source()?, content.translation()?)
            else {
                continue;
            };
            if source.text.value.trim().is_empty()
                || translation.text.value.trim().is_empty()
                || translation.language.as_ref().is_some_and(|language| {
                    language.as_str() != target_language.tag()
                        && language.as_str().parse::<Language>().ok() != Some(target_language)
                })
            {
                continue;
            }
            context.push(TranslationContext {
                source: source.text.value,
                translation: translation.text.value,
            });
        }
    }
    Ok(context)
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        Translator::model(&self.config.model)
    }

    /// A hosted provider only waits on the network; holding the accelerator
    /// for that wait serialized every remote request with the GPU stages.
    fn uses_accelerator(&self) -> bool {
        self.config.model.provider == koharu_translator::Provider::Local
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    async fn load(&self) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let mut targets = Vec::new();
        if let Some(group) = input.scene.page(input.page)?.text_group()? {
            for layer in group.text_layers()? {
                if !input.contains_entity(layer.id())? {
                    continue;
                }
                let content = layer.content()?;
                let Some(source) = content.source()? else {
                    continue;
                };
                if !source.text.value.trim().is_empty() {
                    targets.push((content.id(), source.text.value));
                }
            }
        }
        let (source_language, target_language, content_profile, system_prompt) =
            resolved_options(&self.config, input.translation.as_deref());
        let mut request = TranslationRequest::new(
            targets.iter().map(|(_, source)| source.clone()),
            source_language,
            target_language,
        )
        .with_content_profile(content_profile)
        .with_validator(self.validator.clone())
        .with_context(translation_context(
            &input.scene,
            input.page,
            target_language,
        )?)
        // Opaque identifiers for the usage journal; no text leaves the scene.
        .with_usage_context(koharu_usage::UsageContext {
            project_id: Some(input.scene.project_id().to_string()),
            page_id: Some(input.page.to_string()),
            ..koharu_usage::UsageContext::default()
        });
        if let Some(system_prompt) = system_prompt {
            request = request.with_system_prompt(system_prompt);
        }
        if input.translation.is_none()
            && let Some(instructions) = self.config.instructions.as_deref()
        {
            request = request.with_instructions(instructions);
        }
        if Translator::supports_vision(&self.config.model, &self.config.generation)
            && let Some(image) = input.images.get(&input.scene, input.page, "source").await?
        {
            request = request.with_image(image);
        }
        let (provider, translated) = self
            .translator
            .translate(&self.config.model, self.config.generation, request)
            .await?;
        let language = LanguageTag::new(target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = input.scene.edit_as(generated.clone());
        for (entity, _) in &targets {
            edit.observe::<SourceText>(*entity)?;
            edit.observe::<Translation>(*entity)?;
        }
        for ((entity, source), text) in targets.into_iter().zip(translated) {
            if input
                .scene
                .component::<Translation>(entity)?
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let text = if source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                text
            };
            edit.set(
                entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use koharu_translator::Language;

    #[test]
    fn a_run_profile_selects_its_saved_prompt_over_the_editor_profile() {
        let mut config = TranslationConfig::default();
        config.system_prompts.manga = Some("editor prompt".to_owned());
        config.system_prompts.hentai = Some("adult prompt".to_owned());
        let options = TranslationOptions {
            source_language: Language::ChineseSimplified,
            target_language: Language::Russian,
            content_profile: ContentProfile::Hentai,
        };

        assert_eq!(
            resolved_options(&config, Some(&options)),
            (
                Language::ChineseSimplified,
                Language::Russian,
                ContentProfile::Hentai,
                Some("adult prompt"),
            )
        );
        assert_eq!(
            resolved_options(&config, None),
            (
                Language::Japanese,
                Language::English,
                ContentProfile::Manga,
                Some("editor prompt"),
            )
        );
        config.system_prompts.hentai = Some(" ".to_owned());
        assert_eq!(resolved_options(&config, Some(&options)).3, None);
    }

    #[tokio::test]
    async fn memory_uses_prior_pages_in_order_without_future_or_other_language_translations() {
        use koharu_scene::{At, Edit, PageDraft, Session, TextLayout, TextLayoutKind};

        fn segment(
            edit: &mut Edit,
            page: EntityId,
            source: &str,
            translated: &str,
            tag: &str,
        ) -> EntityId {
            let content = edit.add_text_content(page, At::End).unwrap();
            edit.set(
                content,
                &SourceText {
                    text: Authored::user(source.to_owned()),
                    language: None,
                },
            )
            .unwrap();
            edit.set(
                content,
                &Translation {
                    text: Authored::user(translated.to_owned()),
                    language: Some(LanguageTag::new(tag).unwrap()),
                },
            )
            .unwrap();
            edit.add_text_layer(
                page,
                At::End,
                content,
                &TextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Point,
                },
            )
            .unwrap();
            content
        }

        let mut session = Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let first = edit
            .add_page(PageDraft::new("first", 100.0, 100.0), At::End)
            .unwrap();
        let current = edit
            .add_page(PageDraft::new("current", 100.0, 100.0), At::End)
            .unwrap();
        let future = edit
            .add_page(PageDraft::new("future", 100.0, 100.0), At::End)
            .unwrap();
        let alice = segment(&mut edit, first, "Alice", "Алиса", "ru-RU");
        // Two views of one content entity must not repeat its memory entry.
        edit.add_text_layer(
            first,
            At::End,
            alice,
            &TextLayout {
                origin: Origin::User,
                kind: TextLayoutKind::Point,
            },
        )
        .unwrap();
        segment(&mut edit, first, "unused", "   ", "ru-RU");
        segment(&mut edit, first, "different language", "English", "en-US");
        segment(&mut edit, current, "Bob", "Боб", "ru-RU");
        segment(&mut edit, future, "Eve", "Ева", "ru-RU");
        let snapshot = session
            .commit(edit.finish().unwrap())
            .await
            .unwrap()
            .snapshot;
        assert_eq!(
            translation_context(&snapshot, current, Language::Russian).unwrap(),
            vec![TranslationContext {
                source: "Alice".into(),
                translation: "Алиса".into()
            },]
        );
        assert_eq!(
            translation_context(&snapshot, future, Language::Russian).unwrap(),
            vec![
                TranslationContext {
                    source: "Alice".into(),
                    translation: "Алиса".into()
                },
                TranslationContext {
                    source: "Bob".into(),
                    translation: "Боб".into()
                },
            ]
        );
    }
}
