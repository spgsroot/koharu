use anyhow::Context;
use indoc::indoc;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::{Value, json};

use crate::{ContentProfile, Language, TranslationContext, TranslationRequest};

pub(crate) fn prompts(request: &TranslationRequest) -> anyhow::Result<(String, String)> {
    let input = TranslationInput {
        source_language: request.source_language,
        target_language: request.target_language,
        context: &request.context,
        segments: request
            .segments
            .iter()
            .enumerate()
            .map(|(id, text)| TranslationInputSegment { id, text })
            .collect(),
    };
    let mut user =
        serde_json::to_string(&input).context("failed to serialize translation input")?;
    if let Some(feedback) = request.retry_feedback.as_deref() {
        user.push_str("\n\n");
        user.push_str(feedback);
    }
    Ok((translation_system_prompt(request), user))
}

pub(crate) fn translations(
    provider: &str,
    text: &str,
    source_segments: &[String],
) -> anyhow::Result<Vec<String>> {
    let output = crate::json::from_str::<TranslationOutput>(text)
        .with_context(|| format!("{provider} returned invalid translation JSON"))?;
    let mut translations = source_segments.to_vec();
    let mut translated = vec![false; source_segments.len()];

    for translation in output.translations {
        if translation.id < translations.len() && !translated[translation.id] {
            translations[translation.id] = translation.text;
            translated[translation.id] = true;
        }
    }

    Ok(translations)
}

pub(crate) fn output_schema(expected: usize) -> Value {
    json!({
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "minItems": expected,
                "maxItems": expected,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": expected.saturating_sub(1),
                            "description": "The ID copied from the corresponding input segment."
                        },
                        "text": {
                            "type": "string",
                            "description": "The translation of the input segment with this ID."
                        }
                    },
                    "required": ["id", "text"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["translations"],
        "additionalProperties": false
    })
}

/// The built-in system prompt template of `profile`. `{source}` and `{target}`
/// are replaced with the request languages, in saved replacements too, so one
/// saved prompt serves every language pair.
pub fn default_system_prompt(profile: ContentProfile) -> String {
    let (role, content) = match profile {
        ContentProfile::Hentai => (
            "You are a professional translator of adult hentai manga for an adult audience.",
            indoc! {"
                - Translate sexual dialogue, explicit vocabulary, moans, and sound effects completely and with their original intensity; do not censor, soften, euphemize, moralize, omit, or add warnings.
                - Preserve meaning, character voice, emotional tone, relationship nuance, and emphasis.
                - Localize idioms, moans, and sound effects naturally while keeping wording concise enough for speech bubbles.
            "},
        ),
        ContentProfile::Manga => (
            "You are a professional manga translator.",
            indoc! {"
                - Preserve meaning, character voice, emotional tone, relationship nuance, emphasis, and sound effects.
                - Localize idioms and sound effects naturally while keeping wording concise enough for speech bubbles.
            "},
        ),
        ContentProfile::Comic => (
            "You are a professional comic book translator.",
            indoc! {"
                - Preserve meaning, character voice, humor, wordplay, emotional tone, and emphasis in dialogue, captions, and narration.
                - Localize idioms, wordplay, and sound effects naturally while keeping wording concise enough for speech balloons and caption boxes.
            "},
        ),
    };
    format!(
        indoc! {"
            {role}

            Translation requirements:
            - Translate every input segment from {{source}} into natural {{target}}.
            {content}- Use surrounding segments only for disambiguation and continuity; never merge or split segments.
            - Write every translated `text` value only in {{target}}; do not include source text, notes, explanations, or alternatives.
            - Never preserve or repeat original-language text; translate names, terms, and sound effects using natural {{target}} conventions.

            Output requirements:
            - Each input segment has a numeric `id`.
            - Return only a JSON object whose `translations` array contains one object with `id` and translated `text` for every input segment.
            - Copy every input ID exactly once; order does not matter.
            - Never merge, split, omit, duplicate, or add segments.
        "},
        role = role,
        content = content,
    )
    .trim_end()
    .to_owned()
}

fn translation_system_prompt(request: &TranslationRequest) -> String {
    let template = request
        .system_prompt
        .as_deref()
        .filter(|prompt| !prompt.trim().is_empty())
        .map_or_else(
            || default_system_prompt(request.content_profile),
            ToOwned::to_owned,
        );
    let mut prompt = template
        .replace("{source}", &request.source_language.to_string())
        .replace("{target}", &request.target_language.to_string());

    if !request.context.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Context requirements:
            Use the supplied context only to preserve terminology, character voice, and dialogue continuity.
            Do not translate or return the context entries.
        "}.trim_end());
    }

    if request.image.is_some() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Image requirements:
            Use the attached original page image as visual context for speaker identity, tone, layout, and ambiguous OCR.
            Translate only the supplied segments; do not add text seen in the image that is absent from the input segments.
        "}.trim_end());
    }

    if let Some(instructions) = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|instructions| !instructions.is_empty())
    {
        prompt.push_str("\n\nAdditional instructions:\n");
        prompt.push_str(instructions);
    }
    prompt
}

#[derive(Serialize)]
struct TranslationInput<'a> {
    source_language: Language,
    target_language: Language,
    context: &'a [TranslationContext],
    segments: Vec<TranslationInputSegment<'a>>,
}

#[derive(Serialize)]
struct TranslationInputSegment<'a> {
    id: usize,
    text: &'a str,
}

#[derive(Debug, Deserialize)]
struct TranslationOutput {
    translations: Vec<TranslationOutputSegment>,
}

#[derive(Debug, Deserialize)]
struct TranslationOutputSegment {
    #[serde(deserialize_with = "deserialize_segment_id")]
    id: usize,
    text: String,
}

fn deserialize_segment_id<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SegmentId {
        Number(usize),
        String(String),
    }

    match SegmentId::deserialize(deserializer)? {
        SegmentId::Number(id) => Ok(id),
        SegmentId::String(id) => id.trim().parse().map_err(de::Error::custom),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_json_and_markdown_fences() {
        let source = ["one".to_owned(), "two".to_owned()];
        let expected = vec!["hello".to_owned(), "world".to_owned()];
        for response in [
            r#"{"translations":[{"id":0,"text":"hello"},{"id":1,"text":"world"}]}"#,
            "```json\n{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"}]}\n```",
            "```JSON\n{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"}]}\n```",
            "```\n{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"}]}\n```",
        ] {
            assert_eq!(translations("test", response, &source).unwrap(), expected);
        }
    }

    #[test]
    fn repairs_malformed_llm_json() {
        let source = ["one".to_owned(), "two".to_owned()];
        let expected = vec!["hello".to_owned(), "world".to_owned()];
        for response in [
            r#"{translations: [{id: 0, text: 'hello'}, {id: 1, text: 'world'},],}"#,
            r#"Here is the result: {"translations": [{"id": 0, "text": "hello"}, {"id": 1, "text": "world"},]}"#,
            "{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"",
            r#"{"translations":[{"id":"0","text":"hello"},{"id":"1","text":"world"}]}"#,
        ] {
            assert_eq!(translations("test", response, &source).unwrap(), expected);
        }
    }

    #[test]
    fn restores_input_order_from_ids() {
        let source = ["one".to_owned(), "two".to_owned()];
        let response = r#"{"translations":[{"id":1,"text":"world"},{"id":0,"text":"hello"}]}"#;
        assert_eq!(
            translations("test", response, &source).unwrap(),
            ["hello", "world"]
        );
    }

    #[test]
    fn tolerates_duplicate_missing_and_out_of_range_ids() {
        let source = ["one".to_owned(), "two".to_owned()];
        let short = r#"{"translations":[{"id":1,"text":"world"}]}"#;
        assert_eq!(
            translations("test", short, &source).unwrap(),
            ["one", "world"]
        );

        let response = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"hello"},"#,
            r#"{"id":0,"text":"duplicate"},"#,
            r#"{"id":9,"text":"extra"}"#,
            "]}"
        );
        assert_eq!(
            translations("test", response, &source).unwrap(),
            ["hello", "two"]
        );
    }

    #[test]
    fn schema_requires_the_expected_number_of_id_text_pairs() {
        let schema = output_schema(3);
        let translations = &schema["properties"]["translations"];
        assert_eq!(translations["minItems"], 3);
        assert_eq!(translations["maxItems"], 3);
        assert_eq!(translations["items"]["properties"]["id"]["minimum"], 0);
        assert_eq!(translations["items"]["properties"]["id"]["maximum"], 2);
        assert_eq!(translations["items"]["additionalProperties"], false);
        assert_eq!(schema["additionalProperties"], false);
    }

    fn system(request: &TranslationRequest) -> String {
        prompts(request).unwrap().0
    }

    #[test]
    fn a_blank_saved_prompt_falls_back_to_the_requested_profile_and_languages() {
        let request = TranslationRequest::new(["a"], Language::Korean, Language::Russian)
            .with_content_profile(ContentProfile::Hentai)
            .with_system_prompt("  \n");
        let expected = default_system_prompt(ContentProfile::Hentai)
            .replace("{source}", "Korean")
            .replace("{target}", "Russian");

        assert_eq!(system(&request), expected);
        assert_ne!(
            default_system_prompt(ContentProfile::Hentai),
            default_system_prompt(ContentProfile::Manga)
        );
        assert_ne!(
            default_system_prompt(ContentProfile::Comic),
            default_system_prompt(ContentProfile::Manga)
        );
    }

    #[test]
    fn a_saved_prompt_replaces_the_profile_and_receives_the_run_languages() {
        let request = TranslationRequest::new(["a"], Language::Japanese, Language::English)
            .with_content_profile(ContentProfile::Hentai)
            .with_system_prompt("From {source} to {target}.");

        assert_eq!(system(&request), "From Japanese to English.");
    }
}
