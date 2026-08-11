use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ContractError;

macro_rules! validated_string_id {
    ($name:ident, $field:literal, $docs:literal) => {
        #[doc = $docs]
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Creates a validated identifier.
            ///
            /// # Errors
            ///
            /// Returns [`ContractError`] when the value is empty or contains a
            /// NUL byte.
            pub fn new(value: impl Into<String>) -> Result<Self, ContractError> {
                let value = value.into();
                if value.is_empty() {
                    return Err(ContractError::Empty { field: $field });
                }
                if value.contains('\0') {
                    return Err(ContractError::Invalid {
                        field: $field,
                        value,
                    });
                }
                Ok(Self(value))
            }

            #[must_use]
            /// Returns the validated identifier as text.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

validated_string_id!(BranchName, "branch", "Validated repository branch name.");
validated_string_id!(
    DocumentId,
    "document_id",
    "Deterministic identity of one index document."
);
validated_string_id!(
    RevisionId,
    "revision",
    "Opaque source-control revision identifier."
);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
/// Globally scoped repository identity independent of a checkout path.
pub struct RepositoryId {
    /// Source provider, such as `github` or `local`.
    pub provider: String,
    /// Provider namespace, optionally containing `/`-separated groups.
    pub namespace: String,
    /// Repository name within the namespace.
    pub name: String,
}

impl RepositoryId {
    /// Creates a globally scoped repository identity.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when a component is empty or contains path
    /// separators, a colon, or a NUL byte.
    pub fn new(
        provider: impl Into<String>,
        namespace: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let mut result = Self {
            provider: provider.into(),
            namespace: namespace.into(),
            name: name.into(),
        };
        validate_repository_component("provider", &result.provider)?;
        result.namespace = normalize_namespace(&result.namespace)?;
        validate_repository_component("name", &result.name)?;
        Ok(result)
    }
}

fn normalize_namespace(value: &str) -> Result<String, ContractError> {
    if value.is_empty() {
        return Err(ContractError::Empty { field: "namespace" });
    }
    if value.contains(['\\', ':', '\0'])
        || value.starts_with('/')
        || value.ends_with('/')
        || value
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        return Err(ContractError::Invalid {
            field: "namespace",
            value: value.to_owned(),
        });
    }
    Ok(value.to_owned())
}

impl fmt::Display for RepositoryId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}/{}",
            self.provider, self.namespace, self.name
        )
    }
}

fn validate_repository_component(field: &'static str, value: &str) -> Result<(), ContractError> {
    if value.is_empty() {
        return Err(ContractError::Empty { field });
    }
    if value.contains(['/', '\\', ':', '\0']) {
        return Err(ContractError::Invalid {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

/// Repository-relative path serialized with `/` separators on every platform.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NormalizedPath(String);

impl NormalizedPath {
    /// Normalizes a repository-relative path.
    ///
    /// # Errors
    ///
    /// Rejects empty, absolute, drive-prefixed, parent-traversing, and NUL-byte
    /// paths.
    pub fn new(value: impl AsRef<str>) -> Result<Self, ContractError> {
        let original = value.as_ref();
        if original.is_empty() {
            return Err(ContractError::Empty { field: "path" });
        }
        if original.contains('\0')
            || original.starts_with(['/', '\\'])
            || original.as_bytes().get(1) == Some(&b':')
        {
            return Err(ContractError::Invalid {
                field: "path",
                value: original.to_owned(),
            });
        }

        let replaced = original.replace('\\', "/");
        let mut components = Vec::new();
        for component in replaced.split('/') {
            match component {
                "" | "." => {}
                ".." => {
                    return Err(ContractError::Invalid {
                        field: "path",
                        value: original.to_owned(),
                    });
                }
                other => components.push(other),
            }
        }
        if components.is_empty() {
            return Err(ContractError::Empty { field: "path" });
        }
        Ok(Self(components.join("/")))
    }

    #[must_use]
    /// Returns the normalized repository-relative path.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NormalizedPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Digest algorithm used for source content identity.
pub enum HashAlgorithm {
    /// SHA-256 with a 64-character hexadecimal representation.
    Sha256,
}

impl HashAlgorithm {
    const fn hex_length(self) -> usize {
        match self {
            Self::Sha256 => 64,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
/// Validated, canonical digest of a complete source file.
pub struct ContentHash {
    /// Algorithm used to create the digest.
    pub algorithm: HashAlgorithm,
    /// Lowercase hexadecimal digest.
    pub hex_digest: String,
}

impl ContentHash {
    /// Creates a validated lowercase hexadecimal content hash.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] for a digest of the wrong length or containing
    /// non-hexadecimal characters.
    pub fn new(algorithm: HashAlgorithm, digest: impl Into<String>) -> Result<Self, ContractError> {
        let digest = digest.into().to_ascii_lowercase();
        if digest.len() != algorithm.hex_length() {
            return Err(ContractError::InvalidHexLength {
                field: "content_hash",
                expected: algorithm.hex_length(),
            });
        }
        if !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(ContractError::Invalid {
                field: "content_hash",
                value: digest,
            });
        }
        Ok(Self {
            algorithm,
            hex_digest: digest,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
/// Milliseconds since the Unix epoch.
pub struct UnixMillis(pub u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_normalization_is_platform_independent() {
        assert_eq!(
            NormalizedPath::new(r"src\\parser\\mod.rs")
                .unwrap()
                .as_str(),
            "src/parser/mod.rs"
        );
        assert_eq!(
            NormalizedPath::new("./src//lib.rs").unwrap().as_str(),
            "src/lib.rs"
        );
    }

    #[test]
    fn path_rejects_escape_and_absolute_forms() {
        for invalid in ["../secret", "/etc/passwd", r"C:\\secret", "./"] {
            assert!(NormalizedPath::new(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn content_hash_is_canonicalized() {
        let hash = ContentHash::new(HashAlgorithm::Sha256, "A".repeat(64)).unwrap();
        assert_eq!(hash.hex_digest, "a".repeat(64));
    }

    #[test]
    fn repository_namespace_supports_nested_groups() {
        let repository = RepositoryId::new("gitlab", "platform/search/team", "parser").unwrap();
        assert_eq!(repository.to_string(), "gitlab:platform/search/team/parser");
    }
}
