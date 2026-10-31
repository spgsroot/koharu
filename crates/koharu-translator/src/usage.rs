//! Accounting for translation requests: what each attempt tells the journal.
//!
//! A [`UsageTrace`] rides on the [`TranslationRequest`](crate::TranslationRequest)
//! of one validation attempt. The transport (`remote::send_inference`) and the
//! local backend turn it into [`UsageRecord`]s; the translator later downgrades
//! the latest record when the response turns out to be unusable.

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Instant,
};

use koharu_usage::{RequestSettings, UsageContext, UsageOperation, UsageOutcome, UsageRecord};

use crate::{ModelSelection, TranslationRequest};

#[derive(Clone, Debug)]
pub(crate) struct UsageTrace {
    provider: &'static str,
    model: Option<String>,
    attempt: u32,
    context: UsageContext,
    /// Identifier of the latest record written for this attempt.
    last: Arc<Mutex<Option<String>>>,
}

impl UsageTrace {
    /// The trace for one validation attempt. Only counts and language tags are
    /// taken from the request; its text never reaches the journal.
    pub(crate) fn new(
        selection: &ModelSelection,
        request: &TranslationRequest,
        attempt: usize,
    ) -> Self {
        let mut context = request.usage_context.clone();
        context.source_language = Some(request.source_language.tag().to_owned());
        context.target_language = Some(request.target_language.tag().to_owned());
        context.segments = u32::try_from(request.segments.len()).ok();
        context.images = Some(u32::from(request.image.is_some()));
        context.context_entries = u32::try_from(request.context.len()).ok();
        Self {
            provider: selection.provider.into(),
            model: selection.model.clone(),
            attempt: u32::try_from(attempt).unwrap_or(u32::MAX),
            context,
            last: Arc::default(),
        }
    }

    /// A record for one request of this attempt, stamped as sent now.
    pub(crate) fn begin(&self, settings: RequestSettings, transport_retry: u32) -> UsageRecord {
        let mut record = UsageRecord::new(UsageOperation::Translation, self.provider);
        record.requested_model = self.model.clone();
        record.attempt = self.attempt;
        record.transport_retry = transport_retry;
        record.settings = settings;
        record.context = self.context.clone();
        record
    }

    /// Writes the record and remembers it as the latest of this attempt.
    pub(crate) fn commit(&self, record: &UsageRecord) {
        *self.last.lock().unwrap_or_else(PoisonError::into_inner) = Some(record.id.clone());
        koharu_usage::record(record);
    }

    /// Marks the latest request of this attempt as not having produced usable
    /// output. Requests that already failed keep their own classification.
    pub(crate) fn reject(&self, outcome: UsageOutcome) {
        let id = self
            .last
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(id) = id {
            koharu_usage::amend_outcome(&id, outcome);
        }
    }
}

/// One request in flight. Finishing it writes its record; dropping it
/// unfinished, because the job was stopped, writes it as cancelled.
pub(crate) struct InFlight<'a> {
    trace: &'a UsageTrace,
    record: Option<UsageRecord>,
    started: Instant,
}

impl<'a> InFlight<'a> {
    pub(crate) fn start(
        trace: &'a UsageTrace,
        settings: RequestSettings,
        transport_retry: u32,
    ) -> Self {
        Self {
            trace,
            record: Some(trace.begin(settings, transport_retry)),
            started: Instant::now(),
        }
    }

    pub(crate) fn record(&mut self) -> &mut UsageRecord {
        self.record
            .as_mut()
            .expect("an unfinished request owns its record")
    }

    pub(crate) fn finish(mut self, outcome: UsageOutcome) {
        self.write(outcome);
    }

    fn write(&mut self, outcome: UsageOutcome) {
        if let Some(mut record) = self.record.take() {
            record.outcome = outcome;
            record.latency_ms = latency_ms(self.started);
            self.trace.commit(&record);
        }
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.write(UsageOutcome::Cancelled);
    }
}

/// Elapsed milliseconds as the journal stores them.
pub(crate) fn latency_ms(started: Instant) -> Option<u32> {
    u32::try_from(started.elapsed().as_millis()).ok()
}

#[cfg(test)]
mod tests {
    use crate::{Language, Provider, TranslationContext};

    use super::*;

    fn selection() -> ModelSelection {
        ModelSelection {
            provider: Provider::OpenAi,
            model: Some("gpt-test".to_owned()),
            ..ModelSelection::default()
        }
    }

    #[test]
    fn records_carry_counts_and_language_tags_but_no_text() {
        let request = TranslationRequest::new(
            ["secret first line", "secret second line"],
            Language::Japanese,
            Language::Russian,
        )
        .with_context([TranslationContext::new("earlier source", "earlier target")])
        .with_image(Arc::new(image::DynamicImage::new_rgb8(1, 1)))
        .with_system_prompt("confidential instructions")
        .with_usage_context(UsageContext {
            project_id: Some("project".to_owned()),
            page_id: Some("page".to_owned()),
            ..UsageContext::default()
        });

        let record =
            UsageTrace::new(&selection(), &request, 2).begin(RequestSettings::default(), 1);

        assert_eq!(record.operation, UsageOperation::Translation);
        assert_eq!(record.provider, "openai");
        assert_eq!(record.requested_model.as_deref(), Some("gpt-test"));
        assert_eq!((record.attempt, record.transport_retry), (2, 1));
        assert_eq!(record.context.project_id.as_deref(), Some("project"));
        assert_eq!(record.context.page_id.as_deref(), Some("page"));
        assert_eq!(record.context.job_id, None);
        assert_eq!(
            record.context.source_language.as_deref(),
            Some(Language::Japanese.tag())
        );
        assert_eq!(
            record.context.target_language.as_deref(),
            Some(Language::Russian.tag())
        );
        assert_eq!(record.context.segments, Some(2));
        assert_eq!(record.context.images, Some(1));
        assert_eq!(record.context.context_entries, Some(1));
        let journaled = serde_json::to_string(&record).unwrap();
        for private in ["secret", "earlier", "confidential"] {
            assert!(
                !journaled.contains(private),
                "{private} leaked into the record"
            );
        }
    }

    #[test]
    fn retry_attempts_count_only_the_resent_segments() {
        let mut request =
            TranslationRequest::new(["one", "two", "three"], Language::English, Language::French);
        request.segments.truncate(1);

        let trace = UsageTrace::new(&selection(), &request, 1);

        assert_eq!(
            trace.begin(RequestSettings::default(), 0).context.segments,
            Some(1)
        );
    }
}
