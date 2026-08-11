use std::path::Path;

use cast_core::{ChunkError, LanguageId, LanguageResolution, ParsePolicy, ResolutionMethod};

#[derive(Clone, Debug)]
/// Canonical language selected for a chunking operation.
pub struct ResolvedLanguage {
    /// Canonical language id used to select a compiled grammar.
    pub id: LanguageId,
    /// Public resolution metadata recorded in chunk output.
    pub resolution: LanguageResolution,
}

/// Exact parser package identity included in an index grammar fingerprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrammarVersion {
    /// Canonical language id exposed by the registry.
    pub language: &'static str,
    /// Cargo package that provides the compiled grammar.
    pub package: &'static str,
    /// Exact grammar package version.
    pub version: &'static str,
}

/// Compile-time language registry.
///
/// Rust is always available. Other grammars are controlled by `lang-*` Cargo
/// features, with the broadly used set enabled by default.
#[derive(Clone, Copy, Debug, Default)]
pub struct LanguageRegistry;

impl LanguageRegistry {
    /// Resolves a caller-provided language id or common alias.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::UnsupportedLanguage`] when no compiled grammar
    /// matches and generic fallback is not enabled.
    pub fn resolve_explicit(
        &self,
        requested: &str,
        policy: ParsePolicy,
    ) -> Result<ResolvedLanguage, ChunkError> {
        let normalized = requested.trim().to_ascii_lowercase();
        let canonical = canonical_language(&normalized);
        Self::resolve_canonical(canonical, ResolutionMethod::Explicit, policy)
    }

    /// Resolves a language from a source path.
    ///
    /// Extensions claimed by multiple compiled grammars are errors even when
    /// generic fallback is enabled; callers must supply the language explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::AmbiguousLanguage`] for `.h` when both C and C++
    /// are compiled, or
    /// [`ChunkError::UnsupportedLanguage`] when the extension is not recognized
    /// and generic fallback is not enabled.
    pub fn resolve_path(
        &self,
        path: &Path,
        policy: ParsePolicy,
    ) -> Result<ResolvedLanguage, ChunkError> {
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);

        let canonical = match extension.as_deref() {
            Some("h") => match (cfg!(feature = "lang-c"), cfg!(feature = "lang-cpp")) {
                (true, true) => {
                    return Err(ChunkError::AmbiguousLanguage(format!(
                        "{} may be C or C++; provide an explicit language",
                        path.display()
                    )));
                }
                (true, false) => Some("c"),
                (false, true) => Some("cpp"),
                (false, false) => None,
            },
            Some("rs") => Some("rust"),
            #[cfg(feature = "lang-bash")]
            Some("sh" | "bash") => Some("bash"),
            #[cfg(feature = "lang-c")]
            Some("c") => Some("c"),
            #[cfg(feature = "lang-cpp")]
            Some("cc" | "cpp" | "cxx" | "hh" | "hpp" | "hxx") => Some("cpp"),
            #[cfg(feature = "lang-csharp")]
            Some("cs") => Some("csharp"),
            #[cfg(feature = "lang-go")]
            Some("go") => Some("go"),
            #[cfg(feature = "lang-java")]
            Some("java") => Some("java"),
            #[cfg(feature = "lang-javascript")]
            Some("js" | "jsx" | "mjs" | "cjs") => Some("javascript"),
            #[cfg(feature = "lang-php")]
            Some("php" | "phtml") => Some("php"),
            #[cfg(feature = "lang-python")]
            Some("py" | "pyi" | "pyw") => Some("python"),
            #[cfg(feature = "lang-ruby")]
            Some("rb" | "rake" | "gemspec") => Some("ruby"),
            #[cfg(feature = "lang-typescript")]
            Some("ts") => Some("typescript"),
            #[cfg(feature = "lang-typescript")]
            Some("tsx") => Some("tsx"),
            _ => None,
        };

        match canonical {
            Some(language) => {
                Self::resolve_canonical(language, ResolutionMethod::Extension, policy)
            }
            None if matches!(policy, ParsePolicy::GenericFallback) => {
                Ok(resolved("generic", ResolutionMethod::GenericFallback))
            }
            None => Err(ChunkError::UnsupportedLanguage(path.display().to_string())),
        }
    }

    /// Returns whether an explicit language id resolves to a compiled grammar.
    #[must_use]
    pub fn supports(&self, requested: &str) -> bool {
        let normalized = requested.trim().to_ascii_lowercase();
        self.grammar(&LanguageId::from(canonical_language(&normalized)))
            .is_some()
    }

    /// Canonical language ids with grammars compiled into this build.
    #[must_use]
    pub fn compiled_languages(&self) -> &'static [&'static str] {
        COMPILED_LANGUAGES
    }

    /// Exact runtime and grammar package versions compiled into this build.
    #[must_use]
    pub fn compiled_grammars(&self) -> &'static [GrammarVersion] {
        COMPILED_GRAMMARS
    }

    /// Stable value suitable for an index fingerprint's `grammar_set` field.
    #[must_use]
    pub fn grammar_set_id(&self) -> String {
        let mut identity = String::from("tree-sitter@0.26.11");
        for grammar in COMPILED_GRAMMARS {
            identity.push(';');
            identity.push_str(grammar.language);
            identity.push(':');
            identity.push_str(grammar.package);
            identity.push('@');
            identity.push_str(grammar.version);
        }
        identity
    }

    #[must_use]
    /// Returns a compiled grammar for a canonical language id.
    pub fn grammar(&self, language: &LanguageId) -> Option<tree_sitter::Language> {
        match language.0.as_str() {
            "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
            #[cfg(feature = "lang-bash")]
            "bash" => Some(tree_sitter_bash::LANGUAGE.into()),
            #[cfg(feature = "lang-c")]
            "c" => Some(tree_sitter_c::LANGUAGE.into()),
            #[cfg(feature = "lang-cpp")]
            "cpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
            #[cfg(feature = "lang-csharp")]
            "csharp" => Some(tree_sitter_c_sharp::LANGUAGE.into()),
            #[cfg(feature = "lang-go")]
            "go" => Some(tree_sitter_go::LANGUAGE.into()),
            #[cfg(feature = "lang-java")]
            "java" => Some(tree_sitter_java::LANGUAGE.into()),
            #[cfg(feature = "lang-javascript")]
            "javascript" => Some(tree_sitter_javascript::LANGUAGE.into()),
            #[cfg(feature = "lang-php")]
            "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
            #[cfg(feature = "lang-python")]
            "python" => Some(tree_sitter_python::LANGUAGE.into()),
            #[cfg(feature = "lang-ruby")]
            "ruby" => Some(tree_sitter_ruby::LANGUAGE.into()),
            #[cfg(feature = "lang-typescript")]
            "typescript" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            #[cfg(feature = "lang-typescript")]
            "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
            _ => None,
        }
    }

    fn resolve_canonical(
        canonical: &str,
        method: ResolutionMethod,
        policy: ParsePolicy,
    ) -> Result<ResolvedLanguage, ChunkError> {
        if canonical == "generic" {
            return Ok(resolved("generic", method));
        }
        if LanguageRegistry
            .grammar(&LanguageId::from(canonical))
            .is_some()
        {
            return Ok(resolved(canonical, method));
        }
        if matches!(policy, ParsePolicy::GenericFallback) {
            return Ok(resolved("generic", ResolutionMethod::GenericFallback));
        }
        Err(ChunkError::UnsupportedLanguage(canonical.to_owned()))
    }
}

const COMPILED_LANGUAGES: &[&str] = &[
    #[cfg(feature = "lang-bash")]
    "bash",
    #[cfg(feature = "lang-c")]
    "c",
    #[cfg(feature = "lang-cpp")]
    "cpp",
    #[cfg(feature = "lang-csharp")]
    "csharp",
    #[cfg(feature = "lang-go")]
    "go",
    #[cfg(feature = "lang-java")]
    "java",
    #[cfg(feature = "lang-javascript")]
    "javascript",
    #[cfg(feature = "lang-php")]
    "php",
    #[cfg(feature = "lang-python")]
    "python",
    #[cfg(feature = "lang-ruby")]
    "ruby",
    "rust",
    #[cfg(feature = "lang-typescript")]
    "tsx",
    #[cfg(feature = "lang-typescript")]
    "typescript",
];

const COMPILED_GRAMMARS: &[GrammarVersion] = &[
    #[cfg(feature = "lang-bash")]
    GrammarVersion {
        language: "bash",
        package: "tree-sitter-bash",
        version: "0.25.1",
    },
    #[cfg(feature = "lang-c")]
    GrammarVersion {
        language: "c",
        package: "tree-sitter-c",
        version: "0.24.2",
    },
    #[cfg(feature = "lang-cpp")]
    GrammarVersion {
        language: "cpp",
        package: "tree-sitter-cpp",
        version: "0.23.4",
    },
    #[cfg(feature = "lang-csharp")]
    GrammarVersion {
        language: "csharp",
        package: "tree-sitter-c-sharp",
        version: "0.23.5",
    },
    #[cfg(feature = "lang-go")]
    GrammarVersion {
        language: "go",
        package: "tree-sitter-go",
        version: "0.25.0",
    },
    #[cfg(feature = "lang-java")]
    GrammarVersion {
        language: "java",
        package: "tree-sitter-java",
        version: "0.23.5",
    },
    #[cfg(feature = "lang-javascript")]
    GrammarVersion {
        language: "javascript",
        package: "tree-sitter-javascript",
        version: "0.25.0",
    },
    #[cfg(feature = "lang-php")]
    GrammarVersion {
        language: "php",
        package: "tree-sitter-php",
        version: "0.24.2",
    },
    #[cfg(feature = "lang-python")]
    GrammarVersion {
        language: "python",
        package: "tree-sitter-python",
        version: "0.25.0",
    },
    #[cfg(feature = "lang-ruby")]
    GrammarVersion {
        language: "ruby",
        package: "tree-sitter-ruby",
        version: "0.23.1",
    },
    GrammarVersion {
        language: "rust",
        package: "tree-sitter-rust",
        version: "0.24.2",
    },
    #[cfg(feature = "lang-typescript")]
    GrammarVersion {
        language: "tsx",
        package: "tree-sitter-typescript",
        version: "0.23.2",
    },
    #[cfg(feature = "lang-typescript")]
    GrammarVersion {
        language: "typescript",
        package: "tree-sitter-typescript",
        version: "0.23.2",
    },
];

fn canonical_language(requested: &str) -> &str {
    match requested {
        "rs" => "rust",
        "sh" | "shell" => "bash",
        "c++" | "cc" | "cxx" => "cpp",
        "c#" | "cs" => "csharp",
        "golang" => "go",
        "js" | "jsx" | "node" => "javascript",
        "py" => "python",
        "rb" => "ruby",
        "ts" => "typescript",
        "text" | "txt" => "generic",
        canonical => canonical,
    }
}

fn resolved(language: &str, method: ResolutionMethod) -> ResolvedLanguage {
    let id = LanguageId::from(language);
    ResolvedLanguage {
        resolution: LanguageResolution {
            language: id.clone(),
            method,
        },
        id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "popular-languages")]
    fn detects_popular_languages_by_extension() {
        let cases = [
            ("src/lib.rs", "rust"),
            ("bin/setup.sh", "bash"),
            ("src/main.c", "c"),
            ("src/main.cpp", "cpp"),
            ("src/App.cs", "csharp"),
            ("cmd/main.go", "go"),
            ("src/Main.java", "java"),
            ("web/app.jsx", "javascript"),
            ("public/index.php", "php"),
            ("app/models.py", "python"),
            ("lib/task.rb", "ruby"),
            ("web/app.ts", "typescript"),
            ("web/App.tsx", "tsx"),
        ];

        for (path, expected) in cases {
            let resolved = LanguageRegistry
                .resolve_path(Path::new(path), ParsePolicy::Recover)
                .unwrap();
            assert_eq!(resolved.id.0, expected, "path: {path}");
            assert_eq!(resolved.resolution.method, ResolutionMethod::Extension);
        }
    }

    #[test]
    #[cfg(feature = "popular-languages")]
    fn aliases_resolve_to_canonical_ids() {
        let cases = [
            ("golang", "go"),
            ("py", "python"),
            ("c++", "cpp"),
            ("c#", "csharp"),
        ];
        for (alias, canonical) in cases {
            let resolved = LanguageRegistry
                .resolve_explicit(alias, ParsePolicy::Recover)
                .unwrap();
            assert_eq!(resolved.id.0, canonical);
        }
    }

    #[test]
    #[cfg(all(feature = "lang-c", feature = "lang-cpp"))]
    fn c_header_is_deliberately_ambiguous() {
        let error = LanguageRegistry
            .resolve_path(Path::new("include/value.h"), ParsePolicy::GenericFallback)
            .unwrap_err();

        assert!(matches!(error, ChunkError::AmbiguousLanguage(_)));
    }

    #[test]
    fn unknown_extension_requires_fallback_policy() {
        let error = LanguageRegistry
            .resolve_path(Path::new("README.unknown"), ParsePolicy::Recover)
            .unwrap_err();

        assert!(matches!(error, ChunkError::UnsupportedLanguage(_)));
    }

    #[test]
    fn compiled_language_list_is_sorted_and_unique() {
        assert!(COMPILED_LANGUAGES.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn grammar_set_identity_is_deterministic_and_versioned() {
        let identity = LanguageRegistry.grammar_set_id();
        assert!(identity.starts_with("tree-sitter@0.26.11;"));
        assert!(identity.contains("rust:tree-sitter-rust@0.24.2"));
        assert_eq!(identity, LanguageRegistry.grammar_set_id());
    }
}
