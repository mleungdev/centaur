use std::{env, sync::Arc, time::Duration};

use serde_json::{Value, json};
use thiserror::Error;

const SESSION_TITLE_MAX_SOURCE_CHARS: usize = 4_000;
const SESSION_TITLE_MAX_CHARS: usize = 80;
const SESSION_TITLE_REQUEST_TIMEOUT: Duration = Duration::from_secs(4);
const SESSION_TITLE_REASONING_EFFORT_ENV: &str = "SESSION_TITLE_REASONING_EFFORT";
const SESSION_TITLES_ENABLED_ENV: &str = "SESSION_TITLES_ENABLED";
const SESSION_TITLE_PROVIDER_ENV: &str = "SESSION_TITLE_PROVIDER";
const SESSION_TITLE_MODEL_ENV: &str = "SESSION_TITLE_MODEL";
const SESSION_TITLE_BASE_URL_ENV: &str = "SESSION_TITLE_BASE_URL";
const SESSION_TITLE_API_KEY_ENV_ENV: &str = "SESSION_TITLE_API_KEY_ENV";
/// A five-word commit-style title needs no deliberation, and the request only
/// budgets 24 output tokens. Servers that resolve an absent effort to their
/// highest level spend the whole budget reasoning and return no message, so an
/// explicit low is the safe default rather than leaving it to the server.
const DEFAULT_SESSION_TITLE_REASONING_EFFORT: &str = "low";
const SESSION_TITLE_INSTRUCTIONS: &str = "Generate a short session title for the user's request. Return only the title. Use commit-message style with an imperative verb first, such as Fix, Investigate, Add, Update, Debug, Review, Explain, or Analyze. Keep it to 5 words max; 6-7 words are okay only when needed for a product name. Do not include punctuation, quotes, emoji, markdown, or a trailing period.";
const OPENAI_SESSION_TITLE_MAX_OUTPUT_TOKENS: u32 = 24;
/// Claude models think by default and thinking counts toward `max_tokens`, so
/// the cap leaves room beyond the title itself.
const ANTHROPIC_SESSION_TITLE_MAX_TOKENS: u32 = 1_024;
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SessionTitleProvider {
    /// OpenAI Responses API (`POST {base}/responses`).
    OpenAi,
    /// Anthropic Messages API (`POST {base}/messages`).
    Anthropic,
}

impl SessionTitleProvider {
    fn parse(configured: Option<&str>) -> Option<Self> {
        match configured.map(str::to_ascii_lowercase).as_deref() {
            None | Some("openai") => Some(Self::OpenAi),
            Some("anthropic") => Some(Self::Anthropic),
            Some(_) => None,
        }
    }

    fn default_model(self) -> &'static str {
        match self {
            Self::OpenAi => "gpt-5.4-nano",
            Self::Anthropic => "claude-haiku-5-5",
        }
    }

    fn default_api_key_env(self) -> &'static str {
        match self {
            Self::OpenAi => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
        }
    }
}

/// Where session titles go. Every setting is optional; with none set, titles
/// use OpenAI as before.
#[derive(Debug, PartialEq, Eq)]
struct SessionTitleConfig {
    provider: SessionTitleProvider,
    api_key: String,
    url: String,
    model: String,
}

/// Split from the env reads so the resolution is testable without mutating
/// process-global state. `get` returns an env var's value.
fn resolve_session_title_config(
    get: impl Fn(&str) -> Option<String>,
) -> Option<SessionTitleConfig> {
    let set = |name: &str| {
        get(name)
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    };
    if !session_titles_enabled(set(SESSION_TITLES_ENABLED_ENV).as_deref()) {
        return None;
    }
    let configured_provider = set(SESSION_TITLE_PROVIDER_ENV);
    let Some(provider) = SessionTitleProvider::parse(configured_provider.as_deref()) else {
        tracing::warn!(
            provider = configured_provider.as_deref().unwrap_or_default(),
            "unknown {SESSION_TITLE_PROVIDER_ENV}; generated session titles are off"
        );
        return None;
    };
    let key_env = set(SESSION_TITLE_API_KEY_ENV_ENV)
        .unwrap_or_else(|| provider.default_api_key_env().to_owned());
    let api_key = set(&key_env)?;
    // The literal env var name is the placeholder sandboxes see; never send it.
    if api_key == key_env {
        return None;
    }
    let base_url = set(SESSION_TITLE_BASE_URL_ENV).unwrap_or_else(|| match provider {
        SessionTitleProvider::OpenAi => {
            set("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.openai.com/v1".to_owned())
        }
        SessionTitleProvider::Anthropic => "https://api.anthropic.com/v1".to_owned(),
    });
    let base_url = base_url.trim_end_matches('/');
    let url = match provider {
        SessionTitleProvider::OpenAi => format!("{base_url}/responses"),
        SessionTitleProvider::Anthropic => format!("{base_url}/messages"),
    };
    Some(SessionTitleConfig {
        provider,
        api_key,
        url,
        model: set(SESSION_TITLE_MODEL_ENV).unwrap_or_else(|| provider.default_model().to_owned()),
    })
}

#[derive(Clone)]
pub(crate) struct HttpSessionTitleGenerator {
    provider: SessionTitleProvider,
    api_key: Arc<str>,
    url: Arc<str>,
    model: Arc<str>,
    client: reqwest::Client,
    /// `None` omits the parameter, for servers that reject an unknown field.
    reasoning_effort: Option<Arc<str>>,
}

impl HttpSessionTitleGenerator {
    pub(crate) fn from_env() -> Option<Self> {
        let config = resolve_session_title_config(|name| env::var(name).ok())?;
        Self::new(config, session_title_reasoning_effort())
    }

    fn new(config: SessionTitleConfig, reasoning_effort: Option<Arc<str>>) -> Option<Self> {
        let client = reqwest::Client::builder()
            .timeout(SESSION_TITLE_REQUEST_TIMEOUT)
            .build()
            .ok()?;
        Some(Self {
            provider: config.provider,
            api_key: Arc::from(config.api_key),
            url: Arc::from(config.url),
            model: Arc::from(config.model),
            client,
            reasoning_effort,
        })
    }

    pub(crate) async fn generate(
        &self,
        source: String,
    ) -> Result<String, SessionTitleGenerationError> {
        let input = format!("User request:\n{}", source);
        let request = match self.provider {
            SessionTitleProvider::OpenAi => {
                let mut body = json!({
                    "model": self.model.as_ref(),
                    "instructions": SESSION_TITLE_INSTRUCTIONS,
                    "input": input,
                    "max_output_tokens": OPENAI_SESSION_TITLE_MAX_OUTPUT_TOKENS,
                });
                if let Some(effort) = &self.reasoning_effort {
                    body["reasoning"] = json!({ "effort": effort.as_ref() });
                }
                self.client
                    .post(self.url.as_ref())
                    .bearer_auth(self.api_key.as_ref())
                    .json(&body)
            }
            SessionTitleProvider::Anthropic => {
                let mut body = json!({
                    "model": self.model.as_ref(),
                    "max_tokens": ANTHROPIC_SESSION_TITLE_MAX_TOKENS,
                    "system": SESSION_TITLE_INSTRUCTIONS,
                    "messages": [{ "role": "user", "content": input }],
                });
                if let Some(effort) = &self.reasoning_effort {
                    body["output_config"] = json!({ "effort": effort.as_ref() });
                }
                self.client
                    .post(self.url.as_ref())
                    .header("x-api-key", self.api_key.as_ref())
                    .header("anthropic-version", ANTHROPIC_VERSION)
                    .json(&body)
            }
        };
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(SessionTitleGenerationError::HttpStatus { status, body: text });
        }
        match self.provider {
            SessionTitleProvider::OpenAi => openai_response_output_text(&text),
            SessionTitleProvider::Anthropic => anthropic_message_text(&text),
        }
        .ok_or(SessionTitleGenerationError::MissingOutput)
    }
}

/// Titles are on unless explicitly turned off. This lets a deployment that has
/// `OPENAI_API_KEY` for Codex keep titles off.
fn session_titles_enabled(configured: Option<&str>) -> bool {
    !matches!(
        configured
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Some("false" | "0" | "no" | "off")
    )
}

/// Reasoning effort for the title call: OpenAI's `reasoning.effort`, or
/// Anthropic's `output_config.effort`. An empty value omits the parameter,
/// which is the escape hatch for a server that rejects it outright.
fn session_title_reasoning_effort() -> Option<Arc<str>> {
    resolve_reasoning_effort(env::var(SESSION_TITLE_REASONING_EFFORT_ENV).ok().as_deref())
}

/// Split from the env read so the resolution is testable without mutating
/// process-global state.
fn resolve_reasoning_effort(configured: Option<&str>) -> Option<Arc<str>> {
    match configured.map(str::trim) {
        Some("") => None,
        Some(value) => Some(Arc::from(value.to_owned())),
        None => Some(Arc::from(DEFAULT_SESSION_TITLE_REASONING_EFFORT)),
    }
}

pub(crate) fn session_title_source_from_parts(parts: &[Value]) -> Option<String> {
    let mut text_blocks = Vec::new();
    let mut slack_thread_source = None;
    let mut attachment_names = Vec::new();
    for part in parts {
        match part {
            Value::String(text) => {
                collect_title_source_text(text, &mut text_blocks, &mut slack_thread_source);
            }
            Value::Object(object) => {
                if let Some(text) = object.get("text").and_then(Value::as_str) {
                    collect_title_source_text(text, &mut text_blocks, &mut slack_thread_source);
                }
                for key in ["name", "title", "filename"] {
                    if let Some(name) = object.get(key).and_then(Value::as_str)
                        && let Some(name) = clean_nonempty(name)
                    {
                        attachment_names.push(name.to_owned());
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    let source = slack_thread_source
        .or_else(|| text_blocks.first().cloned())
        .or_else(|| {
            attachment_names
                .first()
                .map(|name| format!("Analyze attachment {name}"))
        })?;
    Some(truncate_chars(&source, SESSION_TITLE_MAX_SOURCE_CHARS))
}

fn collect_title_source_text(
    raw_text: &str,
    text_blocks: &mut Vec<String>,
    slack_thread_source: &mut Option<String>,
) {
    if slack_thread_source.is_none()
        && let Some(text) = slack_thread_context_title_source(raw_text)
        && title_source_has_signal(&text)
    {
        *slack_thread_source = Some(text);
    }
    if is_session_context_text(raw_text) {
        return;
    }
    if let Some(text) = clean_title_source_text(raw_text)
        && title_source_has_signal(&text)
    {
        text_blocks.push(text);
    }
}

fn slack_thread_context_title_source(text: &str) -> Option<String> {
    if !text.trim_start().starts_with("# Slack Thread Context") {
        return None;
    }

    let mut in_first_message = false;
    let mut lines = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "# Current Request" {
            break;
        }
        if !in_first_message {
            if trimmed.starts_with("1. ") && trimmed.ends_with(':') {
                in_first_message = true;
            }
            continue;
        }
        if trimmed.starts_with("2. ") && trimmed.ends_with(':') {
            break;
        }
        if trimmed.is_empty() {
            if lines.is_empty() {
                continue;
            }
            break;
        }
        lines.push(trimmed);
    }

    let text = lines.join(" ");
    clean_title_source_text(&text)
}

fn title_source_has_signal(text: &str) -> bool {
    let normalized = normalize_low_signal_text(text);
    if normalized.is_empty() {
        return false;
    }
    if is_low_signal_phrase(&normalized) {
        return false;
    }

    let words = normalized.split_whitespace().collect::<Vec<_>>();
    if words.iter().all(|word| is_low_signal_word(word)) {
        return false;
    }

    true
}

fn normalize_low_signal_text(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let lowercase = text.to_lowercase();
    let mut chars = lowercase.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == ':' {
            let mut emoji_name = String::new();
            while let Some(next) = chars.peek().copied() {
                chars.next();
                if next == ':' {
                    break;
                }
                if next.is_ascii_alphanumeric() || matches!(next, '_' | '-' | '+') {
                    emoji_name.push(next);
                    continue;
                }
                output.push(' ');
                output.push_str(&emoji_name);
                output.push(next);
                emoji_name.clear();
                break;
            }
            continue;
        }
        if ch.is_alphanumeric() {
            output.push(ch);
        } else {
            output.push(' ');
        }
    }
    output.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_low_signal_phrase(text: &str) -> bool {
    matches!(
        text,
        "hey"
            | "hi"
            | "hello"
            | "yo"
            | "hey bot"
            | "hi bot"
            | "hello bot"
            | "hey ai"
            | "hi ai"
            | "hello ai"
            | "thread"
            | "help"
            | "can you help"
            | "can you help me"
            | "please help"
            | "pls help"
    )
}

fn is_low_signal_word(word: &str) -> bool {
    matches!(
        word,
        "hey" | "hi" | "hello" | "yo" | "bot" | "ai" | "thread" | "please" | "pls"
    )
}

fn clean_title_source_text(text: &str) -> Option<String> {
    let text = strip_slack_user_mentions(text)
        .replace('\r', "\n")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let mut text = clean_nonempty(&text)?.to_owned();
    if text.starts_with('@') {
        text = text
            .char_indices()
            .find(|(_, ch)| ch.is_whitespace())
            .map(|(index, _)| text[index..].trim_start().to_owned())
            .unwrap_or_default();
    }
    clean_nonempty(&text).map(str::to_owned)
}

fn strip_slack_user_mentions(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<@") {
        output.push_str(&rest[..start]);
        let mention = &rest[start + 2..];
        let Some(end) = mention.find('>') else {
            output.push_str(&rest[start..]);
            return output;
        };
        rest = &mention[end + 1..];
    }
    output.push_str(rest);
    output
}

fn is_session_context_text(text: &str) -> bool {
    let text = text.trim_start();
    [
        "# Requester Context",
        "# Slack Session Context",
        "# Slack Thread Context",
        "Earlier Slack thread attachment",
    ]
    .iter()
    .any(|prefix| text.starts_with(prefix))
}

pub(crate) fn sanitize_session_title(title: &str) -> Option<String> {
    let title = title
        .trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '"' | '\'' | '`' | '*' | '_' | '-' | ':' | ';' | ',' | '.'
            )
        })
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let title = clean_nonempty(&title)?;
    let words = title
        .split_whitespace()
        .take(7)
        .map(|word| {
            word.trim_matches(|ch: char| matches!(ch, '"' | '\'' | '`' | ',' | '.' | ':' | ';'))
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if words.is_empty() {
        return None;
    }
    Some(truncate_chars(&words.join(" "), SESSION_TITLE_MAX_CHARS))
}

fn openai_response_output_text(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    if let Some(text) = value.get("output_text").and_then(Value::as_str)
        && clean_nonempty(text).is_some()
    {
        return Some(text.to_owned());
    }
    for output in value.get("output").and_then(Value::as_array)? {
        let Some(content) = output.get("content").and_then(Value::as_array) else {
            continue;
        };
        for item in content {
            if let Some(text) = item.get("text").and_then(Value::as_str)
                && clean_nonempty(text).is_some()
            {
                return Some(text.to_owned());
            }
        }
    }
    None
}

/// First non-empty `text` block of a Messages API response. A response can
/// start with `thinking` blocks, so blocks are matched by type, not position.
fn anthropic_message_text(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    value
        .get("content")?
        .as_array()?
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .find(|text| clean_nonempty(text).is_some())
        .map(str::to_owned)
}

fn clean_nonempty(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty() { None } else { Some(value) }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut truncated = value.chars().take(max_chars).collect::<String>();
    if truncated.ends_with(char::is_whitespace) {
        truncated = truncated.trim_end().to_owned();
    }
    truncated
}

#[derive(Debug, Error)]
pub enum SessionTitleGenerationError {
    #[error("session title response did not include output text")]
    MissingOutput,
    #[error("session title request failed with status {status}: {body}")]
    HttpStatus {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error(transparent)]
    Http(#[from] reqwest::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_title_source_prefers_user_ask_over_slack_context() {
        let parts = vec![
            json!({
                "type": "text",
                "text": "# Requester Context\n\nThe Slack user who prompted this turn is Alice."
            }),
            json!({
                "type": "text",
                "text": "<@U123> please fix the memory leak in the worker"
            }),
        ];

        assert_eq!(
            session_title_source_from_parts(&parts),
            Some("please fix the memory leak in the worker".to_owned())
        );
    }

    #[test]
    fn session_title_source_uses_first_slack_thread_message() {
        let parts = vec![
            json!({
                "type": "text",
                "text": "# Slack Thread Context\n\nEarlier messages in this Slack thread, in chronological order:\n\n1. Alice:\n   Planning to replace the billing export job with a streaming worker because the nightly batch keeps timing out\n\n# Current Request\n\nThe user message follows in the next content block.\n---"
            }),
            json!({
                "type": "text",
                "text": "<@U123> investigate this"
            }),
        ];

        assert_eq!(
            session_title_source_from_parts(&parts),
            Some(
                "Planning to replace the billing export job with a streaming worker because the nightly batch keeps timing out"
                    .to_owned()
            )
        );
    }

    #[test]
    fn session_title_source_skips_low_signal_wakeups() {
        assert_eq!(
            session_title_source_from_parts(&[
                json!({"type": "text", "text": "<@U123> Hey"}),
                json!({"type": "text", "text": ":thread:"}),
            ]),
            None
        );

        assert_eq!(
            session_title_source_from_parts(&[
                json!({"type": "text", "text": "<@U123> Hey"}),
                json!({"type": "text", "text": "Can you investigate queue stalls?"}),
            ]),
            Some("Can you investigate queue stalls?".to_owned())
        );
    }

    #[test]
    fn sanitize_session_title_keeps_model_wording() {
        assert_eq!(
            sanitize_session_title("Memory leak in worker queue needs investigation immediately"),
            Some("Memory leak in worker queue needs investigation".to_owned())
        );
        assert_eq!(
            sanitize_session_title("\"Fix worker memory leak.\""),
            Some("Fix worker memory leak".to_owned())
        );
    }

    #[test]
    fn openai_response_output_text_reads_responses_api_shapes() {
        assert_eq!(
            openai_response_output_text(r#"{"output_text":"Fix worker memory leak"}"#),
            Some("Fix worker memory leak".to_owned())
        );
        assert_eq!(
            openai_response_output_text(
                r#"{"output":[{"content":[{"type":"output_text","text":"Add Tempo Explorer filter"}]}]}"#
            ),
            Some("Add Tempo Explorer filter".to_owned())
        );
    }

    /// The generator reads the effort from the environment, so these assert the
    /// resolution rather than spinning a server: an unset variable must still
    /// send an explicit effort, because leaving it to the server's default is
    /// exactly the failure.
    #[test]
    fn title_reasoning_effort_defaults_to_low() {
        assert_eq!(
            resolve_reasoning_effort(None).as_deref(),
            Some(DEFAULT_SESSION_TITLE_REASONING_EFFORT)
        );
    }

    #[test]
    fn session_titles_default_on_and_can_be_turned_off() {
        assert!(session_titles_enabled(None));
        assert!(session_titles_enabled(Some("true")));
        for off in ["false", "0", "OFF", " no "] {
            assert!(!session_titles_enabled(Some(off)), "{off:?}");
        }
    }

    /// Empty is the escape hatch for a server that rejects the field outright.
    #[test]
    fn title_reasoning_effort_empty_omits_the_parameter() {
        assert!(resolve_reasoning_effort(Some("  ")).is_none());
    }

    fn resolve(vars: &[(&str, &str)]) -> Option<SessionTitleConfig> {
        let vars: std::collections::HashMap<String, String> = vars
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        resolve_session_title_config(|name| vars.get(name).cloned())
    }

    #[test]
    fn session_title_provider_defaults_to_openai() {
        let expected = Some(SessionTitleConfig {
            provider: SessionTitleProvider::OpenAi,
            api_key: "sk-test".to_owned(),
            url: "https://gw.example/v1/responses".to_owned(),
            model: "gpt-5.4-nano".to_owned(),
        });
        let openai = [
            ("OPENAI_API_KEY", "sk-test"),
            ("OPENAI_BASE_URL", "https://gw.example/v1/"),
            ("ANTHROPIC_API_KEY", "anthropic-test"),
        ];
        assert_eq!(resolve(&openai), expected);
        assert_eq!(
            resolve(&[openai[0], openai[1], ("SESSION_TITLE_PROVIDER", " ")]),
            expected
        );
        assert_eq!(resolve(&[("ANTHROPIC_API_KEY", "anthropic-test")]), None);
        assert_eq!(resolve(&[("OPENAI_API_KEY", "OPENAI_API_KEY")]), None);
    }

    #[test]
    fn anthropic_session_titles_use_their_own_key_url_and_model() {
        assert_eq!(
            resolve(&[
                ("SESSION_TITLE_PROVIDER", "anthropic"),
                ("ANTHROPIC_API_KEY", "anthropic-test"),
                ("OPENAI_API_KEY", "sk-test"),
                ("OPENAI_BASE_URL", "https://gw.example/v1"),
            ]),
            Some(SessionTitleConfig {
                provider: SessionTitleProvider::Anthropic,
                api_key: "anthropic-test".to_owned(),
                url: "https://api.anthropic.com/v1/messages".to_owned(),
                model: "claude-haiku-5-5".to_owned(),
            })
        );
        assert_eq!(
            resolve(&[
                ("SESSION_TITLE_PROVIDER", "anthropic"),
                ("SESSION_TITLE_API_KEY_ENV", "TITLE_KEY"),
                ("TITLE_KEY", "title-key"),
                ("SESSION_TITLE_BASE_URL", "https://gateway.example/v1/"),
                ("SESSION_TITLE_MODEL", "small-model"),
            ]),
            Some(SessionTitleConfig {
                provider: SessionTitleProvider::Anthropic,
                api_key: "title-key".to_owned(),
                url: "https://gateway.example/v1/messages".to_owned(),
                model: "small-model".to_owned(),
            })
        );
        // No Anthropic key means no titles, not a fallback to OpenAI.
        assert_eq!(
            resolve(&[
                ("SESSION_TITLE_PROVIDER", "anthropic"),
                ("OPENAI_API_KEY", "sk-test")
            ]),
            None
        );
        assert_eq!(
            resolve(&[
                ("SESSION_TITLE_PROVIDER", "other"),
                ("OPENAI_API_KEY", "sk-test")
            ]),
            None
        );
        assert_eq!(
            resolve(&[
                ("SESSION_TITLE_PROVIDER", "anthropic"),
                ("ANTHROPIC_API_KEY", "anthropic-test"),
                ("SESSION_TITLES_ENABLED", "false"),
            ]),
            None
        );
    }

    #[tokio::test]
    async fn anthropic_session_title_calls_the_messages_api() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            let header_end = loop {
                let read = stream.read(&mut buf).await.unwrap();
                assert!(read > 0, "connection closed before the request headers");
                request.extend_from_slice(&buf[..read]);
                if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                    break index + 4;
                }
            };
            let head = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
            let content_length = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .map(|value| value.trim().parse::<usize>().unwrap())
                .unwrap_or_default();
            while request.len() < header_end + content_length {
                let read = stream.read(&mut buf).await.unwrap();
                assert!(read > 0, "connection closed before the request body");
                request.extend_from_slice(&buf[..read]);
            }
            let body = r#"{"type":"message","role":"assistant","content":[{"type":"thinking","thinking":"","signature":"sig"},{"type":"text","text":"Fix worker memory leak"}],"stop_reason":"end_turn"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            let _ = stream.shutdown().await;
            (
                String::from_utf8_lossy(&request[..header_end]).into_owned(),
                serde_json::from_slice::<Value>(&request[header_end..]).unwrap(),
            )
        });

        let config = resolve(&[
            ("SESSION_TITLE_PROVIDER", "anthropic"),
            ("ANTHROPIC_API_KEY", "anthropic-test"),
            ("SESSION_TITLE_BASE_URL", &base_url),
        ])
        .expect("anthropic config");
        let generator =
            HttpSessionTitleGenerator::new(config, resolve_reasoning_effort(None)).unwrap();
        let title = generator
            .generate("please fix the memory leak in the worker".to_owned())
            .await
            .unwrap();
        assert_eq!(title, "Fix worker memory leak");

        let (head, body) = server.await.unwrap();
        let head_lower = head.to_ascii_lowercase();
        assert!(head.starts_with("POST /v1/messages HTTP/1.1\r\n"), "{head}");
        assert!(
            head_lower.contains("\r\nx-api-key: anthropic-test\r\n"),
            "{head}"
        );
        assert!(
            head_lower.contains("\r\nanthropic-version: 2023-06-01\r\n"),
            "{head}"
        );
        assert!(!head_lower.contains("\r\nauthorization:"), "{head}");
        assert_eq!(
            body,
            json!({
                "model": "claude-haiku-5-5",
                "max_tokens": ANTHROPIC_SESSION_TITLE_MAX_TOKENS,
                "system": SESSION_TITLE_INSTRUCTIONS,
                "messages": [{
                    "role": "user",
                    "content": "User request:\nplease fix the memory leak in the worker",
                }],
                "output_config": { "effort": "low" },
            })
        );
    }

    #[test]
    fn anthropic_message_text_skips_non_text_blocks() {
        assert_eq!(
            anthropic_message_text(
                r#"{"content":[{"type":"thinking","thinking":"","signature":"sig"},{"type":"text","text":" "},{"type":"text","text":"Add Tempo Explorer filter"}]}"#
            ),
            Some("Add Tempo Explorer filter".to_owned())
        );
        assert_eq!(
            anthropic_message_text(r#"{"content":[],"stop_reason":"refusal"}"#),
            None
        );
    }
}
