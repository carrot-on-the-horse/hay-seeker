use crate::SizeError;

/// Measures source slices in a monotonic unit.
///
/// Tokenizers may retokenize the end of a prefix when text is extended, so
/// implementations are not required to be perfectly monotonic. The strict
/// splitter validates the selected boundary before emitting it.
pub trait Sizer: Send + Sync {
    /// Returns the stable serialized name of the sizing strategy.
    fn name(&self) -> &'static str;

    /// Measures a source slice.
    ///
    /// # Errors
    ///
    /// Returns [`SizeError`] when an implementation cannot measure the input.
    fn measure(&self, text: &str) -> Result<usize, SizeError>;
}

#[derive(Debug, Default)]
/// Measures the UTF-8 byte length of source text.
pub struct ByteSizer;

impl Sizer for ByteSizer {
    fn name(&self) -> &'static str {
        "bytes"
    }

    fn measure(&self, text: &str) -> Result<usize, SizeError> {
        Ok(text.len())
    }
}

#[derive(Debug, Default)]
/// Measures whitespace-delimited Unicode words.
pub struct UnicodeWordSizer;

impl Sizer for UnicodeWordSizer {
    fn name(&self) -> &'static str {
        "unicode_words"
    }

    fn measure(&self, text: &str) -> Result<usize, SizeError> {
        Ok(text.split_whitespace().count())
    }
}

#[derive(Debug, Default)]
/// Measures logical lines, excluding an empty line after a trailing newline.
pub struct LineSizer;

impl Sizer for LineSizer {
    fn name(&self) -> &'static str {
        "lines"
    }

    fn measure(&self, text: &str) -> Result<usize, SizeError> {
        if text.is_empty() {
            return Ok(0);
        }

        Ok(text.bytes().filter(|byte| *byte == b'\n').count() + usize::from(!text.ends_with('\n')))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_sizer_counts_utf8_bytes() {
        assert_eq!(ByteSizer.measure("世界").unwrap(), 6);
    }

    #[test]
    fn line_sizer_ignores_trailing_empty_line() {
        assert_eq!(LineSizer.measure("one\ntwo\n").unwrap(), 2);
    }

    #[test]
    fn word_sizer_uses_unicode_whitespace() {
        assert_eq!(UnicodeWordSizer.measure("one\u{2003}two").unwrap(), 2);
    }
}
