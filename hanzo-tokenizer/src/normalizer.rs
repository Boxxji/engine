// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

use std::borrow::Cow;

use regex::{NoExpand, Regex};
use serde::Deserialize;
use serde_json::Value;
use unicode_normalization_alignments::UnicodeNormalization;

use crate::TokenizerError;

#[derive(Debug, Clone)]
pub(crate) enum Normalizer {
    None,
    Nfc,
    Nfkc,
    Nfd,
    Nfkd,
    Lowercase,
    Prepend(String),
    Strip { left: bool, right: bool },
    ReplaceString { pattern: String, content: String },
    ReplaceRegex { regex: Regex, content: String },
    Sequence(Vec<Normalizer>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum NormalizerConfig {
    #[serde(rename = "NFC")]
    Nfc,
    #[serde(rename = "NFKC")]
    Nfkc,
    #[serde(rename = "NFD")]
    Nfd,
    #[serde(rename = "NFKD")]
    Nfkd,
    Lowercase,
    Prepend {
        prepend: String,
    },
    Strip {
        #[serde(default)]
        strip_left: bool,
        #[serde(default)]
        strip_right: bool,
    },
    Replace {
        pattern: ReplacePatternConfig,
        content: String,
    },
    Sequence {
        normalizers: Vec<Value>,
    },
}

#[derive(Deserialize)]
enum ReplacePatternConfig {
    String(String),
    Regex(String),
}

impl Normalizer {
    pub(crate) fn from_value(value: Option<&Value>) -> Result<Self, TokenizerError> {
        let Some(val) = value else {
            return Ok(Self::None);
        };
        if val.is_null() {
            return Ok(Self::None);
        }
        let obj = val.as_object().ok_or_else(|| {
            TokenizerError::InvalidConfig("normalizer must be null or an object".to_string())
        })?;
        let ty = obj.get("type").and_then(Value::as_str).ok_or_else(|| {
            TokenizerError::InvalidConfig("normalizer missing string `type`".to_string())
        })?;

        match ty {
            "NFC" | "NFKC" | "NFD" | "NFKD" | "Lowercase" | "Prepend" | "Strip" | "Replace"
            | "Sequence" => {}
            other => {
                return Err(TokenizerError::Unsupported(format!(
                    "unsupported normalizer type: {other}"
                )));
            }
        }

        let cfg: NormalizerConfig = serde_json::from_value(val.clone())?;
        match cfg {
            NormalizerConfig::Nfc => Ok(Self::Nfc),
            NormalizerConfig::Nfkc => Ok(Self::Nfkc),
            NormalizerConfig::Nfd => Ok(Self::Nfd),
            NormalizerConfig::Nfkd => Ok(Self::Nfkd),
            NormalizerConfig::Lowercase => Ok(Self::Lowercase),
            NormalizerConfig::Prepend { prepend } => {
                if prepend.is_empty() {
                    Ok(Self::None)
                } else {
                    Ok(Self::Prepend(prepend))
                }
            }
            NormalizerConfig::Strip {
                strip_left,
                strip_right,
            } => {
                if !strip_left && !strip_right {
                    Ok(Self::None)
                } else {
                    Ok(Self::Strip {
                        left: strip_left,
                        right: strip_right,
                    })
                }
            }
            NormalizerConfig::Replace { pattern, content } => match pattern {
                ReplacePatternConfig::String(s) => {
                    if s.is_empty() {
                        Ok(Self::None)
                    } else {
                        Ok(Self::ReplaceString {
                            pattern: s,
                            content,
                        })
                    }
                }
                ReplacePatternConfig::Regex(r) => {
                    if contains_unsupported_regex_feature(&r) {
                        return Err(TokenizerError::Unsupported(format!(
                            "unsupported lookaround or backref in Replace normalizer regex: {r}"
                        )));
                    }
                    let regex = Regex::new(&r)?;
                    Ok(Self::ReplaceRegex { regex, content })
                }
            },
            NormalizerConfig::Sequence { normalizers } => {
                let mut flat = Vec::new();
                for child in normalizers {
                    match Self::from_value(Some(&child))? {
                        Self::None => {}
                        Self::Sequence(inner) => flat.extend(inner),
                        other => flat.push(other),
                    }
                }
                match flat.len() {
                    0 => Ok(Self::None),
                    1 => Ok(flat.pop().unwrap_or(Self::None)),
                    _ => Ok(Self::Sequence(flat)),
                }
            }
        }
    }

    pub(crate) fn is_noop(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Whether normalizing `prefix + '\n' + suffix` equals `normalize(prefix + '\n') + normalize(suffix)`
    /// whenever `suffix` begins with a safe non-combining ASCII byte.
    pub(crate) fn is_boundary_safe_for_newline(&self) -> bool {
        match self {
            Self::None | Self::Nfc | Self::Nfkc | Self::Nfd | Self::Nfkd | Self::Lowercase => true,
            Self::Prepend(_)
            | Self::Strip { .. }
            | Self::ReplaceString { .. }
            | Self::ReplaceRegex { .. } => false,
            Self::Sequence(items) => items.iter().all(Self::is_boundary_safe_for_newline),
        }
    }

    pub(crate) fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        match self {
            Self::None => Cow::Borrowed(input),
            Self::Nfc => {
                if input.is_ascii() {
                    Cow::Borrowed(input)
                } else {
                    Cow::Owned(input.nfc().map(|(c, _)| c).collect())
                }
            }
            Self::Nfkc => {
                if input.is_ascii() {
                    Cow::Borrowed(input)
                } else {
                    Cow::Owned(input.nfkc().map(|(c, _)| c).collect())
                }
            }
            Self::Nfd => {
                if input.is_ascii() {
                    Cow::Borrowed(input)
                } else {
                    Cow::Owned(input.nfd().map(|(c, _)| c).collect())
                }
            }
            Self::Nfkd => {
                if input.is_ascii() {
                    Cow::Borrowed(input)
                } else {
                    Cow::Owned(input.nfkd().map(|(c, _)| c).collect())
                }
            }
            Self::Lowercase => {
                if input.is_ascii() {
                    if !input.bytes().any(|b| b.is_ascii_uppercase()) {
                        Cow::Borrowed(input)
                    } else {
                        Cow::Owned(input.to_ascii_lowercase())
                    }
                } else {
                    Cow::Owned(input.chars().flat_map(char::to_lowercase).collect())
                }
            }
            Self::Prepend(prefix) => {
                if input.is_empty() || prefix.is_empty() {
                    Cow::Borrowed(input)
                } else {
                    let mut out = String::with_capacity(prefix.len() + input.len());
                    out.push_str(prefix);
                    out.push_str(input);
                    Cow::Owned(out)
                }
            }
            Self::Strip { left, right } => Cow::Borrowed(strip_str(input, *left, *right)),
            Self::ReplaceString { pattern, content } => {
                if pattern.is_empty() || !input.contains(pattern.as_str()) {
                    Cow::Borrowed(input)
                } else {
                    Cow::Owned(input.replace(pattern.as_str(), content.as_str()))
                }
            }
            Self::ReplaceRegex { regex, content } => {
                regex.replace_all(input, NoExpand(content.as_str()))
            }
            Self::Sequence(items) => {
                let mut current = Cow::Borrowed(input);
                for norm in items {
                    current = match current {
                        Cow::Borrowed(s) => norm.normalize(s),
                        Cow::Owned(owned) => match norm.normalize(&owned) {
                            Cow::Borrowed(sub) => {
                                if sub.len() == owned.len() {
                                    Cow::Owned(owned)
                                } else {
                                    Cow::Owned(sub.to_owned())
                                }
                            }
                            Cow::Owned(new_owned) => Cow::Owned(new_owned),
                        },
                    };
                }
                current
            }
        }
    }
}

fn strip_str(input: &str, left: bool, right: bool) -> &str {
    let mut s = input;
    if left {
        s = s.trim_start_matches(char::is_whitespace);
    }
    if right {
        s = s.trim_end_matches(char::is_whitespace);
    }
    s
}

pub(crate) fn contains_unsupported_regex_feature(pattern: &str) -> bool {
    pattern.contains("(?=")
        || pattern.contains("(?!")
        || pattern.contains("(?<=")
        || pattern.contains("(?<!")
}
