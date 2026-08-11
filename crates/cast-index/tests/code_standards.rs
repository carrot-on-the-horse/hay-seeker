use std::fs;
use std::path::{Path, PathBuf};

use syn::visit::Visit;
use syn::{Attribute, ExprMethodCall, ItemFn, ItemMod};

#[derive(Default)]
struct PanicHelperVisitor {
    violations: Vec<&'static str>,
}

impl<'ast> Visit<'ast> for PanicHelperVisitor {
    fn visit_item_mod(&mut self, item: &'ast ItemMod) {
        if test_only(&item.attrs) {
            return;
        }
        syn::visit::visit_item_mod(self, item);
    }

    fn visit_item_fn(&mut self, item: &'ast ItemFn) {
        if item.sig.ident == "main" || test_only(&item.attrs) {
            return;
        }
        syn::visit::visit_item_fn(self, item);
    }

    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        if expression.method == "unwrap" {
            self.violations.push("unwrap");
        } else if expression.method == "expect" {
            self.violations.push("expect");
        }
        syn::visit::visit_expr_method_call(self, expression);
    }
}

fn test_only(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("test")
            || (attribute.path().is_ident("cfg")
                && attribute
                    .meta
                    .require_list()
                    .is_ok_and(|list| list.tokens.to_string().contains("test")))
    })
}

fn rust_sources(directory: &Path, sources: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
}

#[test]
fn production_code_avoids_unwrap_and_expect() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let crates = workspace.join("crates");
    let mut sources = Vec::new();
    for entry in fs::read_dir(crates).unwrap() {
        let source = entry.unwrap().path().join("src");
        if source.is_dir() {
            rust_sources(&source, &mut sources);
        }
    }
    sources.sort();

    let mut failures = Vec::new();
    for source in sources {
        let contents = fs::read_to_string(&source).unwrap();
        let syntax = syn::parse_file(&contents).unwrap();
        let mut visitor = PanicHelperVisitor::default();
        visitor.visit_file(&syntax);
        if !visitor.violations.is_empty() {
            failures.push(format!(
                "{}: {}",
                source.strip_prefix(workspace).unwrap().display(),
                visitor.violations.join(", ")
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "unwrap/expect calls are forbidden outside tests and main:\n{}",
        failures.join("\n")
    );
}

#[test]
fn library_crates_enforce_public_docs_and_publish_examples() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let crates = workspace.join("crates");
    let mut failures = Vec::new();

    for entry in fs::read_dir(crates).unwrap() {
        let library = entry.unwrap().path().join("src/lib.rs");
        if !library.is_file() {
            continue;
        }
        let contents = fs::read_to_string(&library).unwrap();
        let crate_name = library
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        if !contents.contains("#![deny(missing_docs)]") {
            failures.push(format!("{crate_name}: missing deny(missing_docs)"));
        }
        if !contents.contains("//! ```") {
            failures.push(format!("{crate_name}: missing compiling crate example"));
        }
    }

    assert!(
        failures.is_empty(),
        "library documentation contract failed:\n{}",
        failures.join("\n")
    );
}
