use std::time::{Instant, SystemTime, UNIX_EPOCH};

use koharu_usage::{TokenUsage, UsageOutcome, UsageRecord};

/// One HTTP inference attempt, including early returns and dropped stream futures.
/// Only allowlisted metadata is retained; provider messages never enter the journal.
pub(crate) struct Inference {
    pub(crate) entry: UsageRecord,
    started: Instant,
}

impl Inference {
    pub(crate) fn new(mut entry: UsageRecord) -> Self {
        entry.started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|time| time.as_millis().try_into().ok())
            .unwrap_or(entry.started_at);
        Self {
            entry,
            started: Instant::now(),
        }
    }

    pub(crate) fn sent(
        &mut self,
        result: anyhow::Result<reqwest::Response>,
        control: &crate::Control,
    ) -> anyhow::Result<reqwest::Response> {
        let response = result.inspect_err(|error| {
            self.entry.outcome = if control.is_cancelled() {
                UsageOutcome::Cancelled
            } else if error
                .downcast_ref::<reqwest::Error>()
                .is_some_and(reqwest::Error::is_timeout)
            {
                UsageOutcome::Timeout
            } else {
                UsageOutcome::NetworkError
            };
        })?;
        self.headers(&response);
        Ok(response)
    }

    pub(crate) fn retry_entry(&self) -> UsageRecord {
        let mut retry = self.entry.clone();
        let identity = UsageRecord::new(
            koharu_usage::UsageOperation::Agent,
            self.entry.provider.clone(),
        );
        retry.id = identity.id;
        retry.started_at = identity.started_at;
        retry.outcome = UsageOutcome::Incomplete;
        retry.error_code = None;
        retry.transport_retry = retry.transport_retry.saturating_add(1);
        retry.http_status = None;
        retry.provider_request_id = None;
        retry.provider_response_id = None;
        retry.resolved_model = None;
        retry.tokens = TokenUsage::default();
        retry.cost = None;
        retry.latency_ms = None;
        retry
    }

    pub(crate) fn headers(&mut self, response: &reqwest::Response) {
        self.entry.http_status = Some(response.status().as_u16());
        self.entry.provider_request_id = response
            .headers()
            .get("x-request-id")
            .or_else(|| response.headers().get("request-id"))
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
    }
}

impl Drop for Inference {
    fn drop(&mut self) {
        self.entry.latency_ms = self.started.elapsed().as_millis().try_into().ok();
        koharu_usage::record(&self.entry);
    }
}
