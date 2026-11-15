// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

use std::borrow::Cow;

use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

use crate::normalizer::contains_unsupported_regex_feature;
use crate::unicode_lut::{
    unicode_props, CLS_LL, CLS_LM, CLS_LO, CLS_LT, CLS_LU, CLS_M, CLS_MASK, CLS_N, CLS_P, CLS_S,
    FLAG_HAN, FLAG_WS,
};
use crate::TokenizerError;

const ASCII_LETTER: u8 = 1 << 0;
const ASCII_UPPER: u8 = 1 << 1;
const ASCII_LOWER: u8 = 1 << 2;
const ASCII_DIGIT: u8 = 1 << 3;
const ASCII_PUNCT: u8 = 1 << 4;
const ASCII_WS: u8 = 1 << 5;
const ASCII_CR_LF: u8 = 1 << 6;

const ASCII_LUT: [u8; 128] = build_ascii_lut();

const fn build_ascii_lut() -> [u8; 128] {
    let mut lut = [0u8; 128];
    let mut i = 0usize;
    while i < 128 {
        let b = i as u8;
        let mut f = 0u8;
        if b >= b'A' && b <= b'Z' {
            f |= ASCII_LETTER | ASCII_UPPER;
        } else if b >= b'a' && b <= b'z' {
            f |= ASCII_LETTER | ASCII_LOWER;
        } else if b >= b'0' && b <= b'9' {
            f |= ASCII_DIGIT;
        } else if (b >= 0x21 && b <= 0x2F)
            || (b >= 0x3A && b <= 0x40)
            || (b >= 0x5B && b <= 0x60)
            || (b >= 0x7B && b <= 0x7E)
        {
            f |= ASCII_PUNCT;
        }
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == 0x0B || b == 0x0C {
            f |= ASCII_WS;
        }
        if b == b'\r' || b == b'\n' {
            f |= ASCII_CR_LF;
        }
        lut[i] = f;
        i += 1;
    }
    lut
}

#[inline(always)]
fn is_han(c: char) -> bool {
    (c as u32) >= 0x80 && (unicode_props(c) & FLAG_HAN) != 0
}

#[inline(always)]
fn decode_char(bytes: &[u8], pos: usize) -> (char, usize) {
    let b = bytes[pos];
    if b < 0x80 {
        (b as char, 1)
    } else {
        // Safety: `bytes` comes from valid UTF-8 `&str` and `pos` is on a char boundary.
        let s = unsafe { std::str::from_utf8_unchecked(&bytes[pos..]) };
        let ch = s.chars().next().unwrap_or('\0');
        (ch, ch.len_utf8())
    }
}

#[inline(always)]
fn is_ws_char(c: char) -> bool {
    let u = c as u32;
    if u < 128 {
        (ASCII_LUT[u as usize] & ASCII_WS) != 0
    } else {
        (unicode_props(c) & FLAG_WS) != 0
    }
}

#[inline(always)]
fn is_letter_char(c: char) -> bool {
    let u = c as u32;
    if u < 128 {
        (ASCII_LUT[u as usize] & ASCII_LETTER) != 0
    } else {
        (unicode_props(c) & CLS_MASK).wrapping_sub(CLS_LU) <= (CLS_LO - CLS_LU)
    }
}

#[inline(always)]
fn is_number_char(c: char) -> bool {
    let u = c as u32;
    if u < 128 {
        (ASCII_LUT[u as usize] & ASCII_DIGIT) != 0
    } else {
        (unicode_props(c) & CLS_MASK) == CLS_N
    }
}

#[inline(always)]
fn is_mark_char(c: char) -> bool {
    if (c as u32) < 128 {
        false
    } else {
        (unicode_props(c) & CLS_MASK) == CLS_M
    }
}

#[inline(always)]
fn is_punct_char(c: char) -> bool {
    let u = c as u32;
    if u < 128 {
        (ASCII_LUT[u as usize] & ASCII_PUNCT) != 0
    } else {
        (unicode_props(c) & CLS_MASK) == CLS_P
    }
}

#[inline(always)]
fn is_symbol_char(c: char) -> bool {
    let u = c as u32;
    if u < 128 {
        matches!(
            u as u8,
            b'$' | b'+' | b'<' | b'=' | b'>' | b'^' | b'`' | b'|' | b'~'
        )
    } else {
        (unicode_props(c) & CLS_MASK) == CLS_S
    }
}

#[inline(always)]
fn is_o200k_upper(c: char, exclude_han: bool) -> bool {
    let u = c as u32;
    if u < 128 {
        return (ASCII_LUT[u as usize] & ASCII_UPPER) != 0;
    }
    let p = unicode_props(c);
    if exclude_han && (p & FLAG_HAN) != 0 {
        return false;
    }
    const MASK: u16 = (1 << CLS_LU) | (1 << CLS_LT) | (1 << CLS_LM) | (1 << CLS_LO) | (1 << CLS_M);
    ((1u16 << (p & CLS_MASK)) & MASK) != 0
}

#[inline(always)]
fn is_o200k_lower(c: char, exclude_han: bool) -> bool {
    let u = c as u32;
    if u < 128 {
        return (ASCII_LUT[u as usize] & ASCII_LOWER) != 0;
    }
    let p = unicode_props(c);
    if exclude_han && (p & FLAG_HAN) != 0 {
        return false;
    }
    const MASK: u16 = (1 << CLS_LL) | (1 << CLS_LM) | (1 << CLS_LO) | (1 << CLS_M);
    ((1u16 << (p & CLS_MASK)) & MASK) != 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScannerKind {
    Cl100k {
        digits_max: usize,
        marks_in_word: bool,
    },
    O200k {
        split_han: bool,
        allow_contractions: bool,
        digits_max: usize,
        slash_in_nl: bool,
    },
    DeepSeekDigits,
    DeepSeekCjk,
    DeepSeekMain,
    Gpt2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub(crate) enum SplitBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

#[derive(Debug, Clone)]
pub(crate) enum GenericPattern {
    Literal(String),
    Standard(Regex),
    WsLookahead { regex: Regex, has_ws_fallback: bool },
}

#[derive(Debug, Clone)]
pub(crate) struct GenericSplit {
    pattern: GenericPattern,
    behavior: SplitBehavior,
    invert: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum PreTokenizer {
    None,
    Digits {
        individual: bool,
    },
    ByteLevel {
        add_prefix_space: bool,
        use_regex: bool,
    },
    SplitFast(ScannerKind),
    SplitGeneric(GenericSplit),
    Sequence(Vec<PreTokenizer>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum PreTokenizerConfig {
    Digits {
        #[serde(default)]
        individual_digits: bool,
    },
    ByteLevel {
        #[serde(default = "default_true")]
        add_prefix_space: bool,
        #[serde(default = "default_true")]
        #[allow(dead_code)]
        trim_offsets: bool,
        #[serde(default = "default_true")]
        use_regex: bool,
    },
    Split {
        pattern: SplitPatternConfig,
        behavior: SplitBehavior,
        #[serde(default)]
        invert: bool,
    },
    Sequence {
        pretokenizers: Vec<Value>,
    },
}

#[derive(Deserialize)]
enum SplitPatternConfig {
    String(String),
    Regex(String),
}

fn default_true() -> bool {
    true
}

impl PreTokenizer {
    pub(crate) fn from_value(value: Option<&Value>) -> Result<Self, TokenizerError> {
        let Some(val) = value else {
            return Ok(Self::None);
        };
        if val.is_null() {
            return Ok(Self::None);
        }
        let obj = val.as_object().ok_or_else(|| {
            TokenizerError::InvalidConfig("pre_tokenizer must be null or an object".to_string())
        })?;
        let ty = obj.get("type").and_then(Value::as_str).ok_or_else(|| {
            TokenizerError::InvalidConfig("pre_tokenizer missing string `type`".to_string())
        })?;

        match ty {
            "Digits" | "ByteLevel" | "Split" | "Sequence" => {}
            other => {
                return Err(TokenizerError::Unsupported(format!(
                    "unsupported pre_tokenizer type: {other}"
                )));
            }
        }

        let cfg: PreTokenizerConfig = serde_json::from_value(val.clone())?;
        match cfg {
            PreTokenizerConfig::Digits { individual_digits } => Ok(Self::Digits {
                individual: individual_digits,
            }),
            PreTokenizerConfig::ByteLevel {
                add_prefix_space,
                trim_offsets: _,
                use_regex,
            } => Ok(Self::ByteLevel {
                add_prefix_space,
                use_regex,
            }),
            PreTokenizerConfig::Split {
                pattern,
                behavior,
                invert,
            } => match pattern {
                SplitPatternConfig::String(s) => Ok(Self::SplitGeneric(GenericSplit {
                    pattern: GenericPattern::Literal(s),
                    behavior,
                    invert,
                })),
                SplitPatternConfig::Regex(r) => {
                    if !invert && behavior == SplitBehavior::Isolated {
                        if let Some(kind) = recognize_fast_scanner(&r) {
                            return Ok(Self::SplitFast(kind));
                        }
                    }
                    Ok(Self::SplitGeneric(GenericSplit::from_regex(
                        &r, behavior, invert,
                    )?))
                }
            },
            PreTokenizerConfig::Sequence { pretokenizers } => {
                let mut flat = Vec::new();
                for child in pretokenizers {
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

    pub(crate) fn is_pure_byte_level_no_prefix(&self) -> bool {
        match self {
            Self::ByteLevel {
                add_prefix_space: false,
                use_regex: false,
            } => true,
            Self::Sequence(items) => {
                let Some(last) = items.last() else {
                    return false;
                };
                if !matches!(
                    last,
                    Self::ByteLevel {
                        add_prefix_space: false,
                        use_regex: false
                    }
                ) {
                    return false;
                }
                items[..items.len() - 1]
                    .iter()
                    .all(|pt| !matches!(pt, Self::ByteLevel { .. }))
            }
            _ => false,
        }
    }

    /// True when pre-tokenizing `prefix + '\n' + suffix` (where `suffix` begins with a safe
    /// ASCII printable char other than `'\''` or `'/'`) produces the exact concatenation of
    /// pre-tokenizing `prefix + '\n'` and pre-tokenizing `suffix`.
    pub(crate) fn is_boundary_safe_for_newline(&self) -> bool {
        match self {
            Self::SplitFast(kind) => !matches!(kind, ScannerKind::Gpt2),
            Self::ByteLevel {
                add_prefix_space: false,
                use_regex: false,
            } => true,
            Self::Sequence(items) => items.iter().all(Self::is_boundary_safe_for_newline),
            _ => false,
        }
    }

    /// Emit raw UTF-8 pieces for the pure-Split prefix of a ByteLevel pipeline
    /// (excluding the final `ByteLevel { add_prefix_space: false, use_regex: false }`).
    pub(crate) fn split_for_byte_level<'a, F>(&self, input: &'a str, mut emit: F)
    where
        F: FnMut(&'a str, usize),
    {
        match self {
            Self::ByteLevel {
                add_prefix_space: false,
                use_regex: false,
            } => {
                if !input.is_empty() {
                    emit(input, input.len());
                }
            }
            Self::Sequence(items) => {
                let chain = &items[..items.len() - 1];
                run_chain_slices(chain, input, 0, &mut emit);
            }
            _ => {}
        }
    }

    /// General pre-tokenization path yielding owned/borrowed pieces ready for BPE.
    pub(crate) fn pre_tokenize_general(&self, input: &str) -> Vec<(String, usize)> {
        if input.is_empty() {
            return Vec::new();
        }
        let mut pieces: Vec<(Cow<'_, str>, usize)> = vec![(Cow::Borrowed(input), input.len())];
        self.apply_general(&mut pieces);
        pieces
            .into_iter()
            .filter(|(s, _)| !s.is_empty())
            .map(|(s, end)| (s.into_owned(), end))
            .collect()
    }

    fn apply_general<'a>(&self, pieces: &mut Vec<(Cow<'a, str>, usize)>) {
        match self {
            Self::None => {}
            Self::Digits { individual } => {
                let mut next = Vec::with_capacity(pieces.len());
                for (piece, base_end) in pieces.drain(..) {
                    let base_start = base_end - piece.len();
                    split_digits(&piece, *individual, |sub, end| {
                        next.push((Cow::Owned(sub.to_owned()), base_start + end));
                    });
                }
                *pieces = next;
            }
            Self::SplitFast(kind) => {
                let mut next = Vec::with_capacity(pieces.len());
                for (piece, base_end) in pieces.drain(..) {
                    let base_start = base_end - piece.len();
                    match piece {
                        Cow::Borrowed(s) => {
                            scan_fast(*kind, s, |sub, end| {
                                next.push((Cow::Borrowed(sub), base_start + end));
                            });
                        }
                        Cow::Owned(s) => {
                            scan_fast(*kind, &s, |sub, end| {
                                next.push((Cow::Owned(sub.to_owned()), base_start + end));
                            });
                        }
                    }
                }
                *pieces = next;
            }
            Self::SplitGeneric(gen) => {
                let mut next = Vec::with_capacity(pieces.len());
                for (piece, base_end) in pieces.drain(..) {
                    let base_start = base_end - piece.len();
                    match piece {
                        Cow::Borrowed(s) => {
                            gen.split(s, |sub, end| {
                                next.push((Cow::Borrowed(sub), base_start + end));
                            });
                        }
                        Cow::Owned(s) => {
                            gen.split(&s, |sub, end| {
                                next.push((Cow::Owned(sub.to_owned()), base_start + end));
                            });
                        }
                    }
                }
                *pieces = next;
            }
            Self::ByteLevel {
                add_prefix_space,
                use_regex,
            } => {
                let mut next = Vec::with_capacity(pieces.len());
                for (piece, base_end) in pieces.drain(..) {
                    let base_start = base_end - piece.len();
                    let prefixed: Cow<'_, str> =
                        if *add_prefix_space && !piece.is_empty() && !piece.starts_with(' ') {
                            let mut s = String::with_capacity(piece.len() + 1);
                            s.push(' ');
                            s.push_str(&piece);
                            Cow::Owned(s)
                        } else {
                            piece
                        };
                    if *use_regex {
                        scan_gpt2(&prefixed, &mut |sub, end| {
                            let mapped = crate::bpe::encode_byte_level_string(sub.as_bytes());
                            let orig_end = if *add_prefix_space && end > prefixed.len() {
                                base_end
                            } else {
                                (base_start + end).min(base_end)
                            };
                            next.push((Cow::Owned(mapped), orig_end));
                        });
                    } else {
                        let mapped = crate::bpe::encode_byte_level_string(prefixed.as_bytes());
                        next.push((Cow::Owned(mapped), base_end));
                    }
                }
                *pieces = next;
            }
            Self::Sequence(items) => {
                for step in items {
                    step.apply_general(pieces);
                }
            }
        }
    }
}

fn run_chain_slices<'a, F>(chain: &[PreTokenizer], slice: &'a str, base_start: usize, emit: &mut F)
where
    F: FnMut(&'a str, usize),
{
    if slice.is_empty() {
        return;
    }
    let Some((first, rest)) = chain.split_first() else {
        emit(slice, base_start + slice.len());
        return;
    };
    match first {
        PreTokenizer::None => run_chain_slices(rest, slice, base_start, emit),
        PreTokenizer::Digits { individual } => {
            split_digits(slice, *individual, |sub, end| {
                let sub_start = base_start + end - sub.len();
                run_chain_slices(rest, sub, sub_start, emit);
            });
        }
        PreTokenizer::SplitFast(kind) => {
            scan_fast(*kind, slice, |sub, end| {
                let sub_start = base_start + end - sub.len();
                run_chain_slices(rest, sub, sub_start, emit);
            });
        }
        PreTokenizer::SplitGeneric(gen) => {
            gen.split(slice, |sub, end| {
                let sub_start = base_start + end - sub.len();
                run_chain_slices(rest, sub, sub_start, emit);
            });
        }
        _ => {}
    }
}

impl GenericSplit {
    fn from_regex(
        pattern: &str,
        behavior: SplitBehavior,
        invert: bool,
    ) -> Result<Self, TokenizerError> {
        let ws_two = r"|\s+(?!\S)|\s+";
        let ws_one = r"|\s+(?!\S)";
        if let Some(prefix) = pattern.strip_suffix(ws_two) {
            if contains_unsupported_regex_feature(prefix) {
                return Err(TokenizerError::Unsupported(format!(
                    "unsupported lookaround in Split regex: {pattern}"
                )));
            }
            let combined = format!("(?:{prefix})|(?P<__ws_tail>\\s+)");
            let regex = compile_regex_or_unsupported(&combined, pattern)?;
            return Ok(Self {
                pattern: GenericPattern::WsLookahead {
                    regex,
                    has_ws_fallback: true,
                },
                behavior,
                invert,
            });
        }
        if let Some(prefix) = pattern.strip_suffix(ws_one) {
            if contains_unsupported_regex_feature(prefix) {
                return Err(TokenizerError::Unsupported(format!(
                    "unsupported lookaround in Split regex: {pattern}"
                )));
            }
            let combined = format!("(?:{prefix})|(?P<__ws_tail>\\s+)");
            let regex = compile_regex_or_unsupported(&combined, pattern)?;
            return Ok(Self {
                pattern: GenericPattern::WsLookahead {
                    regex,
                    has_ws_fallback: false,
                },
                behavior,
                invert,
            });
        }

        if contains_unsupported_regex_feature(pattern) {
            return Err(TokenizerError::Unsupported(format!(
                "unsupported lookaround in Split regex: {pattern}"
            )));
        }
        let regex = compile_regex_or_unsupported(pattern, pattern)?;
        Ok(Self {
            pattern: GenericPattern::Standard(regex),
            behavior,
            invert,
        })
    }

    fn split<'a, F>(&self, text: &'a str, mut emit: F)
    where
        F: FnMut(&'a str, usize),
    {
        if text.is_empty() {
            return;
        }
        let mut matches: Vec<((usize, usize), bool)> = match &self.pattern {
            GenericPattern::Literal(pat) => find_literal_matches(text, pat),
            GenericPattern::Standard(re) => find_regex_matches(text, re),
            GenericPattern::WsLookahead {
                regex,
                has_ws_fallback,
            } => find_ws_lookahead_matches(text, regex, *has_ws_fallback),
        };

        if self.invert {
            for item in &mut matches {
                item.1 = !item.1;
            }
        }

        emit_split_behavior(text, &matches, self.behavior, &mut emit);
    }
}

fn compile_regex_or_unsupported(compiled: &str, original: &str) -> Result<Regex, TokenizerError> {
    match Regex::new(compiled) {
        Ok(re) => Ok(re),
        Err(err) => {
            if original.contains("?+")
                || original.contains("++")
                || original.contains("*+")
                || original.contains("}+")
                || original.contains("(?")
            {
                Err(TokenizerError::Unsupported(format!(
                    "unsupported regex construct in Split pattern: {original}"
                )))
            } else {
                Err(TokenizerError::Regex(err))
            }
        }
    }
}

fn find_literal_matches(text: &str, pat: &str) -> Vec<((usize, usize), bool)> {
    if pat.is_empty() {
        return vec![((0, text.len()), false)];
    }
    let mut prev = 0usize;
    let mut out = Vec::new();
    for (start, m) in text.match_indices(pat) {
        if prev != start {
            out.push(((prev, start), false));
        }
        let end = start + m.len();
        out.push(((start, end), true));
        prev = end;
    }
    if prev != text.len() {
        out.push(((prev, text.len()), false));
    }
    out
}

fn find_regex_matches(text: &str, re: &Regex) -> Vec<((usize, usize), bool)> {
    let mut prev = 0usize;
    let mut out = Vec::new();
    for m in re.find_iter(text) {
        if prev != m.start() {
            out.push(((prev, m.start()), false));
        }
        out.push(((m.start(), m.end()), true));
        prev = m.end();
    }
    if prev != text.len() {
        out.push(((prev, text.len()), false));
    }
    out
}

fn find_ws_lookahead_matches(
    text: &str,
    re: &Regex,
    has_ws_fallback: bool,
) -> Vec<((usize, usize), bool)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut prev_match_end = 0usize;

    while pos <= text.len() {
        let Some(caps) = re.captures_at(text, pos) else {
            break;
        };
        let Some(full) = caps.get(0) else {
            break;
        };
        let m_start = full.start();
        let m_end = full.end();

        if let Some(ws) = caps.name("__ws_tail") {
            let ws_start = ws.start();
            let ws_end = ws.end();
            if ws_end == text.len() {
                if prev_match_end < ws_start {
                    out.push(((prev_match_end, ws_start), false));
                }
                out.push(((ws_start, ws_end), true));
                prev_match_end = ws_end;
                break;
            }
            // Trim trailing whitespace char so (?!\S) succeeds before it.
            let trimmed_end = text[ws_start..ws_end]
                .char_indices()
                .last()
                .map_or(ws_start, |(rel, _)| ws_start + rel);
            if trimmed_end > ws_start {
                if prev_match_end < ws_start {
                    out.push(((prev_match_end, ws_start), false));
                }
                out.push(((ws_start, trimmed_end), true));
                prev_match_end = trimmed_end;
                pos = trimmed_end;
            } else if has_ws_fallback {
                if prev_match_end < ws_start {
                    out.push(((prev_match_end, ws_start), false));
                }
                out.push(((ws_start, ws_end), true));
                prev_match_end = ws_end;
                pos = ws_end;
            } else {
                pos = ws_end;
            }
        } else {
            if prev_match_end < m_start {
                out.push(((prev_match_end, m_start), false));
            }
            out.push(((m_start, m_end), true));
            prev_match_end = m_end;
            if m_end == m_start {
                if pos == text.len() {
                    break;
                }
                let (_, len) = decode_char(text.as_bytes(), pos);
                pos += len;
            } else {
                pos = m_end;
            }
        }
    }

    if prev_match_end < text.len() {
        out.push(((prev_match_end, text.len()), false));
    }
    if out.is_empty() {
        out.push(((0, 0), false));
    }
    out
}

fn emit_split_behavior<'a, F>(
    text: &'a str,
    matches: &[((usize, usize), bool)],
    behavior: SplitBehavior,
    emit: &mut F,
) where
    F: FnMut(&'a str, usize),
{
    let mut final_splits: Vec<((usize, usize), bool)> = Vec::with_capacity(matches.len());
    match behavior {
        SplitBehavior::Isolated => {
            for &(offsets, _) in matches {
                final_splits.push((offsets, false));
            }
        }
        SplitBehavior::Removed => {
            final_splits.extend_from_slice(matches);
        }
        SplitBehavior::Contiguous => {
            let mut prev_match = false;
            for &(offsets, is_match) in matches {
                if is_match == prev_match {
                    if let Some(((_, end), _)) = final_splits.last_mut() {
                        *end = offsets.1;
                    } else {
                        final_splits.push((offsets, false));
                    }
                } else {
                    final_splits.push((offsets, false));
                }
                prev_match = is_match;
            }
        }
        SplitBehavior::MergedWithPrevious => {
            let mut prev_match = false;
            for &(offsets, is_match) in matches {
                if is_match && !prev_match {
                    if let Some(((_, end), _)) = final_splits.last_mut() {
                        *end = offsets.1;
                    } else {
                        final_splits.push((offsets, false));
                    }
                } else {
                    final_splits.push((offsets, false));
                }
                prev_match = is_match;
            }
        }
        SplitBehavior::MergedWithNext => {
            let mut prev_match = false;
            for &(offsets, is_match) in matches.iter().rev() {
                if is_match && !prev_match {
                    if let Some(((start, _), _)) = final_splits.last_mut() {
                        *start = offsets.0;
                    } else {
                        final_splits.push((offsets, false));
                    }
                } else {
                    final_splits.push((offsets, false));
                }
                prev_match = is_match;
            }
            final_splits.reverse();
        }
    }

    for ((start, end), remove) in final_splits {
        if !remove && start < end {
            emit(&text[start..end], end);
        }
    }
}

fn split_digits<'a, F>(text: &'a str, individual: bool, mut emit: F)
where
    F: FnMut(&'a str, usize),
{
    if text.is_empty() {
        return;
    }
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut non_digit_start: Option<usize> = None;

    while pos < len {
        let (ch, clen) = decode_char(bytes, pos);
        if ch.is_numeric() {
            if let Some(start) = non_digit_start.take() {
                emit(&text[start..pos], pos);
            }
            if individual {
                emit(&text[pos..pos + clen], pos + clen);
                pos += clen;
            } else {
                let d_start = pos;
                pos += clen;
                while pos < len {
                    let (nc, nlen) = decode_char(bytes, pos);
                    if !nc.is_numeric() {
                        break;
                    }
                    pos += nlen;
                }
                emit(&text[d_start..pos], pos);
            }
        } else {
            if non_digit_start.is_none() {
                non_digit_start = Some(pos);
            }
            pos += clen;
        }
    }

    if let Some(start) = non_digit_start {
        emit(&text[start..len], len);
    }
}

pub(crate) fn recognize_fast_scanner(pattern: &str) -> Option<ScannerKind> {
    match pattern {
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        | r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+" => {
            Some(ScannerKind::Cl100k {
                digits_max: 1,
                marks_in_word: false,
            })
        }
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        | r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+" => {
            Some(ScannerKind::Cl100k {
                digits_max: 3,
                marks_in_word: false,
            })
        }
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
            Some(ScannerKind::Cl100k {
                digits_max: 1,
                marks_in_word: true,
            })
        }
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
            Some(ScannerKind::Cl100k {
                digits_max: 3,
                marks_in_word: true,
            })
        }
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
            Some(ScannerKind::O200k {
                split_han: false,
                allow_contractions: true,
                digits_max: 3,
                slash_in_nl: true,
            })
        }
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
            Some(ScannerKind::O200k {
                split_han: false,
                allow_contractions: true,
                digits_max: 3,
                slash_in_nl: false,
            })
        }
        r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
            Some(ScannerKind::O200k {
                split_han: true,
                allow_contractions: true,
                digits_max: 3,
                slash_in_nl: false,
            })
        }
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+" => {
            Some(ScannerKind::O200k {
                split_han: false,
                allow_contractions: false,
                digits_max: 1,
                slash_in_nl: true,
            })
        }
        r"\p{N}{1,3}" => Some(ScannerKind::DeepSeekDigits),
        r"[一-龥぀-ゟ゠-ヿ]+" => Some(ScannerKind::DeepSeekCjk),
        r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##
        | concat!(
            r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^"##,
            "\r\n",
            r##"\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+["##,
            "\r\n",
            r##"]*|\s*["##,
            "\r\n",
            r##"]+|\s+(?!\S)|\s+"##
        ) => Some(ScannerKind::DeepSeekMain),
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+" => {
            Some(ScannerKind::Gpt2)
        }
        _ => None,
    }
}

#[inline]
fn scan_fast<'a, F>(kind: ScannerKind, text: &'a str, mut emit: F)
where
    F: FnMut(&'a str, usize),
{
    match kind {
        ScannerKind::Cl100k {
            digits_max,
            marks_in_word,
        } => scan_cl100k(text, digits_max, marks_in_word, &mut emit),
        ScannerKind::O200k {
            split_han,
            allow_contractions,
            digits_max,
            slash_in_nl,
        } => scan_o200k(
            text,
            split_han,
            allow_contractions,
            digits_max,
            slash_in_nl,
            &mut emit,
        ),
        ScannerKind::DeepSeekDigits => scan_deepseek_digits(text, &mut emit),
        ScannerKind::DeepSeekCjk => scan_deepseek_cjk(text, &mut emit),
        ScannerKind::DeepSeekMain => scan_deepseek_main(text, &mut emit),
        ScannerKind::Gpt2 => scan_gpt2(text, &mut emit),
    }
}

#[inline(always)]
fn match_contraction_ci(bytes: &[u8], pos: usize) -> usize {
    if pos + 1 >= bytes.len() || bytes[pos] != b'\'' {
        return 0;
    }
    let b1 = bytes[pos + 1] | 0x20;
    if b1 == b's' || b1 == b't' || b1 == b'm' || b1 == b'd' {
        return 2;
    }
    if pos + 2 < bytes.len() {
        let b2 = bytes[pos + 2] | 0x20;
        if (b1 == b'r' && b2 == b'e') || (b1 == b'v' && b2 == b'e') || (b1 == b'l' && b2 == b'l') {
            return 3;
        }
    }
    0
}

#[inline(always)]
fn match_contraction_lower(bytes: &[u8], pos: usize) -> usize {
    if pos + 1 >= bytes.len() || bytes[pos] != b'\'' {
        return 0;
    }
    let b1 = bytes[pos + 1];
    if b1 == b's' || b1 == b't' || b1 == b'm' || b1 == b'd' {
        return 2;
    }
    if pos + 2 < bytes.len() {
        let b2 = bytes[pos + 2];
        if (b1 == b'r' && b2 == b'e') || (b1 == b'v' && b2 == b'e') || (b1 == b'l' && b2 == b'l') {
            return 3;
        }
    }
    0
}

#[inline]
fn scan_ws_run(bytes: &[u8], pos: usize) -> usize {
    let len = bytes.len();
    let mut cur = pos;
    let mut last_nl_end: Option<usize> = None;
    let mut last_ws_start = pos;

    while cur < len {
        let b = bytes[cur];
        if b < 128 {
            let f = ASCII_LUT[b as usize];
            if (f & ASCII_WS) == 0 {
                break;
            }
            last_ws_start = cur;
            cur += 1;
            if (f & ASCII_CR_LF) != 0 {
                last_nl_end = Some(cur);
            }
        } else {
            let (c, clen) = decode_char(bytes, cur);
            if !c.is_whitespace() {
                break;
            }
            last_ws_start = cur;
            cur += clen;
        }
    }

    if let Some(nl_end) = last_nl_end {
        nl_end
    } else if cur == len || last_ws_start == pos {
        cur
    } else {
        last_ws_start
    }
}

fn scan_cl100k<'a, F>(text: &'a str, digits_max: usize, marks_in_word: bool, emit: &mut F)
where
    F: FnMut(&'a str, usize),
{
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;

    let is_word = |c: char| -> bool { is_letter_char(c) || (marks_in_word && is_mark_char(c)) };
    let is_punct_body = |c: char| -> bool { !is_ws_char(c) && !is_word(c) && !is_number_char(c) };

    while pos < len {
        // ASCII fast path for words and space+word
        let b0 = bytes[pos];
        if b0 < 128 {
            let f0 = ASCII_LUT[b0 as usize];
            if (f0 & ASCII_LETTER) != 0 {
                let mut end = pos + 1;
                while end < len {
                    let b = bytes[end];
                    if b < 128 {
                        if (ASCII_LUT[b as usize] & ASCII_LETTER) == 0 {
                            break;
                        }
                        end += 1;
                    } else {
                        let (c, clen) = decode_char(bytes, end);
                        if !is_word(c) {
                            break;
                        }
                        end += clen;
                    }
                }
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
            if b0 == b'\'' {
                let clen = match_contraction_ci(bytes, pos);
                if clen > 0 {
                    let end = pos + clen;
                    emit(&text[pos..end], end);
                    pos = end;
                    continue;
                }
            }
            if (f0 & (ASCII_CR_LF | ASCII_DIGIT)) == 0 && pos + 1 < len {
                let b1 = bytes[pos + 1];
                if b1 < 128 && (ASCII_LUT[b1 as usize] & ASCII_LETTER) != 0 {
                    let mut end = pos + 2;
                    while end < len {
                        let b = bytes[end];
                        if b < 128 {
                            if (ASCII_LUT[b as usize] & ASCII_LETTER) == 0 {
                                break;
                            }
                            end += 1;
                        } else {
                            let (c, clen) = decode_char(bytes, end);
                            if !is_word(c) {
                                break;
                            }
                            end += clen;
                        }
                    }
                    emit(&text[pos..end], end);
                    pos = end;
                    continue;
                }
            }
        }

        let (ch, len0) = decode_char(bytes, pos);
        if ch == '\'' {
            let clen = match_contraction_ci(bytes, pos);
            if clen > 0 {
                let end = pos + clen;
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        if is_word(ch) {
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_word(nc) {
                    break;
                }
                end += nlen;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }

        if ch != '\r' && ch != '\n' && !is_number_char(ch) && pos + len0 < len {
            let (nc, nlen) = decode_char(bytes, pos + len0);
            if is_word(nc) {
                let mut end = pos + len0 + nlen;
                while end < len {
                    let (mc, mlen) = decode_char(bytes, end);
                    if !is_word(mc) {
                        break;
                    }
                    end += mlen;
                }
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        if is_number_char(ch) {
            let mut end = pos + len0;
            let mut count = 1usize;
            while count < digits_max && end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_number_char(nc) {
                    break;
                }
                end += nlen;
                count += 1;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }

        if is_punct_body(ch) {
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_punct_body(nc) {
                    break;
                }
                end += nlen;
            }
            while end < len && (bytes[end] == b'\r' || bytes[end] == b'\n') {
                end += 1;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }

        if ch == ' ' && pos + 1 < len {
            let (nc, nlen) = decode_char(bytes, pos + 1);
            if is_punct_body(nc) {
                let mut end = pos + 1 + nlen;
                while end < len {
                    let (mc, mlen) = decode_char(bytes, end);
                    if !is_punct_body(mc) {
                        break;
                    }
                    end += mlen;
                }
                while end < len && (bytes[end] == b'\r' || bytes[end] == b'\n') {
                    end += 1;
                }
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        let end = scan_ws_run(bytes, pos);
        emit(&text[pos..end], end);
        pos = end;
    }
}

/// Match `U* L+` starting at `start` in `bytes`. Returns `None` if no match.
#[inline]
fn match_o200k_u_star_l_plus(bytes: &[u8], start: usize, exclude_han: bool) -> Option<usize> {
    let len = bytes.len();
    let mut cur = start;
    let mut last_in_lower_end: Option<usize> = None;

    while cur < len {
        let b = bytes[cur];
        if b < 128 {
            let f = ASCII_LUT[b as usize];
            if (f & ASCII_UPPER) != 0 {
                cur += 1;
            } else if (f & ASCII_LOWER) != 0 {
                cur += 1;
                while cur < len {
                    let nb = bytes[cur];
                    if nb < 128 {
                        if (ASCII_LUT[nb as usize] & ASCII_LOWER) == 0 {
                            break;
                        }
                        cur += 1;
                    } else {
                        let (nc, nlen) = decode_char(bytes, cur);
                        if !is_o200k_lower(nc, exclude_han) {
                            break;
                        }
                        cur += nlen;
                    }
                }
                return Some(cur);
            } else {
                break;
            }
        } else {
            let (c, clen) = decode_char(bytes, cur);
            let u = is_o200k_upper(c, exclude_han);
            let l = is_o200k_lower(c, exclude_han);
            if !u && !l {
                break;
            }
            if l && !u {
                cur += clen;
                while cur < len {
                    let (nc, nlen) = decode_char(bytes, cur);
                    if !is_o200k_lower(nc, exclude_han) {
                        break;
                    }
                    cur += nlen;
                }
                return Some(cur);
            }
            cur += clen;
            if l {
                last_in_lower_end = Some(cur);
            }
        }
    }

    last_in_lower_end
}

/// Match `U+ L*` starting at `start` in `bytes`. Returns `None` if no match.
#[inline]
fn match_o200k_u_plus_l_star(bytes: &[u8], start: usize, exclude_han: bool) -> Option<usize> {
    let len = bytes.len();
    if start >= len {
        return None;
    }
    let (c0, len0) = decode_char(bytes, start);
    if !is_o200k_upper(c0, exclude_han) {
        return None;
    }
    let mut cur = start + len0;
    while cur < len {
        let (c, clen) = decode_char(bytes, cur);
        if !is_o200k_upper(c, exclude_han) {
            break;
        }
        cur += clen;
    }
    while cur < len {
        let (c, clen) = decode_char(bytes, cur);
        if !is_o200k_lower(c, exclude_han) {
            break;
        }
        cur += clen;
    }
    Some(cur)
}

fn scan_o200k<'a, F>(
    text: &'a str,
    split_han: bool,
    allow_contractions: bool,
    digits_max: usize,
    slash_in_nl: bool,
    emit: &mut F,
) where
    F: FnMut(&'a str, usize),
{
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;

    while pos < len {
        let (ch, len0) = decode_char(bytes, pos);

        if split_han && is_han(ch) {
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_han(nc) {
                    break;
                }
                end += nlen;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }

        let is_prefix = ch != '\r' && ch != '\n' && !is_letter_char(ch) && !is_number_char(ch);

        // Branch 1: [^\r\n\p{L}\p{N}]? U* L+ contr?
        let mut matched_end: Option<usize> = None;
        if is_prefix {
            if let Some(end) = match_o200k_u_star_l_plus(bytes, pos + len0, split_han) {
                matched_end = Some(end);
            }
        }
        if matched_end.is_none() && (is_o200k_upper(ch, split_han) || is_o200k_lower(ch, split_han))
        {
            matched_end = match_o200k_u_star_l_plus(bytes, pos, split_han);
        }

        // Branch 2: [^\r\n\p{L}\p{N}]? U+ L* contr?
        if matched_end.is_none() && is_prefix {
            matched_end = match_o200k_u_plus_l_star(bytes, pos + len0, split_han);
        }
        if matched_end.is_none() && is_o200k_upper(ch, split_han) {
            matched_end = match_o200k_u_plus_l_star(bytes, pos, split_han);
        }

        if let Some(mut end) = matched_end {
            if allow_contractions {
                end += match_contraction_ci(bytes, end);
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }

        if is_number_char(ch) {
            let mut end = pos + len0;
            let mut count = 1usize;
            while count < digits_max && end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_number_char(nc) {
                    break;
                }
                end += nlen;
                count += 1;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }

        let is_punct_body =
            |c: char| -> bool { !is_ws_char(c) && !is_letter_char(c) && !is_number_char(c) };
        if is_punct_body(ch) {
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_punct_body(nc) {
                    break;
                }
                end += nlen;
            }
            while end < len
                && (bytes[end] == b'\r'
                    || bytes[end] == b'\n'
                    || (slash_in_nl && bytes[end] == b'/'))
            {
                end += 1;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }

        if ch == ' ' && pos + 1 < len {
            let (nc, nlen) = decode_char(bytes, pos + 1);
            if is_punct_body(nc) {
                let mut end = pos + 1 + nlen;
                while end < len {
                    let (mc, mlen) = decode_char(bytes, end);
                    if !is_punct_body(mc) {
                        break;
                    }
                    end += mlen;
                }
                while end < len
                    && (bytes[end] == b'\r'
                        || bytes[end] == b'\n'
                        || (slash_in_nl && bytes[end] == b'/'))
                {
                    end += 1;
                }
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        let end = scan_ws_run(bytes, pos);
        emit(&text[pos..end], end);
        pos = end;
    }
}

fn scan_deepseek_digits<'a, F>(text: &'a str, emit: &mut F)
where
    F: FnMut(&'a str, usize),
{
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut gap_start: Option<usize> = None;

    while pos < len {
        let (ch, len0) = decode_char(bytes, pos);
        if is_number_char(ch) {
            if let Some(gs) = gap_start.take() {
                emit(&text[gs..pos], pos);
            }
            let mut end = pos + len0;
            let mut count = 1usize;
            while count < 3 && end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_number_char(nc) {
                    break;
                }
                end += nlen;
                count += 1;
            }
            emit(&text[pos..end], end);
            pos = end;
        } else {
            if gap_start.is_none() {
                gap_start = Some(pos);
            }
            pos += len0;
        }
    }
    if let Some(gs) = gap_start {
        emit(&text[gs..len], len);
    }
}

#[inline(always)]
fn is_ds_cjk(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FA5).contains(&cp)
        || (0x3040..=0x309F).contains(&cp)
        || (0x30A0..=0x30FF).contains(&cp)
}

fn scan_deepseek_cjk<'a, F>(text: &'a str, emit: &mut F)
where
    F: FnMut(&'a str, usize),
{
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut gap_start: Option<usize> = None;

    while pos < len {
        let (ch, len0) = decode_char(bytes, pos);
        if is_ds_cjk(ch) {
            if let Some(gs) = gap_start.take() {
                emit(&text[gs..pos], pos);
            }
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_ds_cjk(nc) {
                    break;
                }
                end += nlen;
            }
            emit(&text[pos..end], end);
            pos = end;
        } else {
            if gap_start.is_none() {
                gap_start = Some(pos);
            }
            pos += len0;
        }
    }
    if let Some(gs) = gap_start {
        emit(&text[gs..len], len);
    }
}

fn scan_deepseek_main<'a, F>(text: &'a str, emit: &mut F)
where
    F: FnMut(&'a str, usize),
{
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut gap_start: Option<usize> = None;

    let is_ds_word = |c: char| -> bool { is_letter_char(c) || is_mark_char(c) };
    let is_ds_prefix = |c: char| -> bool {
        c != '\r' && c != '\n' && !is_letter_char(c) && !is_punct_char(c) && !is_symbol_char(c)
    };
    let is_ds_ps = |c: char| -> bool { is_punct_char(c) || is_symbol_char(c) };

    while pos < len {
        let (ch, len0) = decode_char(bytes, pos);

        // Branch 1: [ASCII punct][A-Za-z]+
        if ch.is_ascii()
            && (ASCII_LUT[ch as usize] & ASCII_PUNCT) != 0
            && pos + 1 < len
            && bytes[pos + 1].is_ascii_alphabetic()
        {
            if let Some(gs) = gap_start.take() {
                emit(&text[gs..pos], pos);
            }
            let mut end = pos + 2;
            while end < len && bytes[end].is_ascii_alphabetic() {
                end += 1;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }

        // Branch 2: [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
        if is_ds_word(ch) {
            if let Some(gs) = gap_start.take() {
                emit(&text[gs..pos], pos);
            }
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_ds_word(nc) {
                    break;
                }
                end += nlen;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }
        if is_ds_prefix(ch) && pos + len0 < len {
            let (nc, nlen) = decode_char(bytes, pos + len0);
            if is_ds_word(nc) {
                if let Some(gs) = gap_start.take() {
                    emit(&text[gs..pos], pos);
                }
                let mut end = pos + len0 + nlen;
                while end < len {
                    let (mc, mlen) = decode_char(bytes, end);
                    if !is_ds_word(mc) {
                        break;
                    }
                    end += mlen;
                }
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        // Branch 3: " ?[\p{P}\p{S}]+[\r\n]*"
        if is_ds_ps(ch) {
            if let Some(gs) = gap_start.take() {
                emit(&text[gs..pos], pos);
            }
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_ds_ps(nc) {
                    break;
                }
                end += nlen;
            }
            while end < len && (bytes[end] == b'\r' || bytes[end] == b'\n') {
                end += 1;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }
        if ch == ' ' && pos + 1 < len {
            let (nc, nlen) = decode_char(bytes, pos + 1);
            if is_ds_ps(nc) {
                if let Some(gs) = gap_start.take() {
                    emit(&text[gs..pos], pos);
                }
                let mut end = pos + 1 + nlen;
                while end < len {
                    let (mc, mlen) = decode_char(bytes, end);
                    if !is_ds_ps(mc) {
                        break;
                    }
                    end += mlen;
                }
                while end < len && (bytes[end] == b'\r' || bytes[end] == b'\n') {
                    end += 1;
                }
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        // Branch 4/5/6: \s*[\r\n]+|\s+(?!\S)|\s+
        if is_ws_char(ch) {
            if let Some(gs) = gap_start.take() {
                emit(&text[gs..pos], pos);
            }
            let end = scan_ws_run(bytes, pos);
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }

        if gap_start.is_none() {
            gap_start = Some(pos);
        }
        pos += len0;
    }

    if let Some(gs) = gap_start {
        emit(&text[gs..len], len);
    }
}

fn scan_gpt2<'a, F>(text: &'a str, emit: &mut F)
where
    F: FnMut(&'a str, usize),
{
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;

    let is_punct_body =
        |c: char| -> bool { !is_ws_char(c) && !is_letter_char(c) && !is_number_char(c) };

    while pos < len {
        let (ch, len0) = decode_char(bytes, pos);

        if ch == '\'' {
            let clen = match_contraction_lower(bytes, pos);
            if clen > 0 {
                let end = pos + clen;
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        if is_letter_char(ch) {
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_letter_char(nc) {
                    break;
                }
                end += nlen;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }
        if ch == ' ' && pos + 1 < len {
            let (nc, nlen) = decode_char(bytes, pos + 1);
            if is_letter_char(nc) {
                let mut end = pos + 1 + nlen;
                while end < len {
                    let (mc, mlen) = decode_char(bytes, end);
                    if !is_letter_char(mc) {
                        break;
                    }
                    end += mlen;
                }
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        if is_number_char(ch) {
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_number_char(nc) {
                    break;
                }
                end += nlen;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }
        if ch == ' ' && pos + 1 < len {
            let (nc, nlen) = decode_char(bytes, pos + 1);
            if is_number_char(nc) {
                let mut end = pos + 1 + nlen;
                while end < len {
                    let (mc, mlen) = decode_char(bytes, end);
                    if !is_number_char(mc) {
                        break;
                    }
                    end += mlen;
                }
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        if is_punct_body(ch) {
            let mut end = pos + len0;
            while end < len {
                let (nc, nlen) = decode_char(bytes, end);
                if !is_punct_body(nc) {
                    break;
                }
                end += nlen;
            }
            emit(&text[pos..end], end);
            pos = end;
            continue;
        }
        if ch == ' ' && pos + 1 < len {
            let (nc, nlen) = decode_char(bytes, pos + 1);
            if is_punct_body(nc) {
                let mut end = pos + 1 + nlen;
                while end < len {
                    let (mc, mlen) = decode_char(bytes, end);
                    if !is_punct_body(mc) {
                        break;
                    }
                    end += mlen;
                }
                emit(&text[pos..end], end);
                pos = end;
                continue;
            }
        }

        // \s+(?!\S)|\s+
        let mut end = pos + len0;
        let mut last_ws_start = pos;
        while end < len {
            let (nc, nlen) = decode_char(bytes, end);
            if !is_ws_char(nc) {
                break;
            }
            last_ws_start = end;
            end += nlen;
        }
        let chosen = if end == len || last_ws_start == pos {
            end
        } else {
            last_ws_start
        };
        emit(&text[pos..chosen], chosen);
        pos = chosen;
    }
}
