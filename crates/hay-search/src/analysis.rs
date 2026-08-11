use std::collections::BTreeMap;

/// Analyzes prose and code identifiers into deterministic BM25 terms.
///
/// The analyzer retains each meaningful lowercase alphanumeric token and
/// additionally emits camel-case components and a conservative inflection
/// stem. Underscores and punctuation split tokens, and common English
/// stopwords are removed, so code identifiers and prose share useful terms
/// without rare stopwords dominating small corpora.
#[must_use]
pub fn analyze_code_terms(text: &str) -> BTreeMap<String, u32> {
    let mut terms = BTreeMap::new();
    for token in text.split(|character: char| !character.is_alphanumeric()) {
        if token.is_empty() {
            continue;
        }
        let lowercase = token.to_lowercase();
        add_analyzed_term(&mut terms, &lowercase);
        for part in camel_parts(token) {
            let part = part.to_lowercase();
            if part != lowercase {
                add_analyzed_term(&mut terms, &part);
            }
        }
    }
    terms
}

fn camel_parts(token: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let characters = token.char_indices().collect::<Vec<_>>();
    for index in 1..characters.len() {
        let (_, previous) = characters[index - 1];
        let (offset, current) = characters[index];
        let next = characters.get(index + 1).map(|(_, value)| *value);
        let boundary = (previous.is_lowercase() || previous.is_numeric()) && current.is_uppercase()
            || previous.is_uppercase()
                && current.is_uppercase()
                && next.is_some_and(char::is_lowercase);
        if boundary {
            parts.push(&token[start..offset]);
            start = offset;
        }
    }
    if start > 0 {
        parts.push(&token[start..]);
    }
    parts
}

fn add_term(terms: &mut BTreeMap<String, u32>, term: &str) {
    if !term.is_empty() {
        let count = terms.entry(term.to_owned()).or_default();
        *count = count.saturating_add(1);
    }
}

fn add_analyzed_term(terms: &mut BTreeMap<String, u32>, term: &str) {
    if is_stopword(term) {
        return;
    }
    add_term(terms, term);
    if let Some(stem) = light_stem(term) {
        add_term(terms, stem);
    }
}

fn light_stem(term: &str) -> Option<&str> {
    for suffix in ["ization", "ation", "ated", "ingly", "edly", "ing", "ed"] {
        if let Some(stem) = term.strip_suffix(suffix)
            && stem.len() >= 4
        {
            return Some(stem);
        }
    }
    for suffix in ["ies", "es", "s"] {
        if let Some(stem) = term.strip_suffix(suffix)
            && stem.len() >= 4
        {
            return Some(stem);
        }
    }
    None
}

fn is_stopword(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "how"
            | "in"
            | "into"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "to"
            | "was"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "why"
            | "with"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_identifier_and_camel_components() {
        let terms = analyze_code_terms("HTTPServer http_server parseJSON2Value");

        assert!(terms.contains_key("httpserver"));
        assert!(terms.contains_key("http"));
        assert!(terms.contains_key("server"));
        assert!(terms.contains_key("parsejson2value"));
        assert!(terms.contains_key("json2"));
        assert!(terms.contains_key("value"));
    }

    #[test]
    fn removes_stopwords_and_aligns_common_inflections() {
        let terms = analyze_code_terms("where is validation validated");

        assert!(!terms.contains_key("where"));
        assert!(!terms.contains_key("is"));
        assert_eq!(terms.get("valid"), Some(&2));
    }
}
