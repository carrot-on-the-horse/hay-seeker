use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

use cast_core::LanguageId;
use cast_index::{DocumentId, NormalizedPath};
use hay_duckdb::DuckDbIndex;
use hay_search::{IndexManifest, SearchDocument};
use tempfile::tempdir;

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn document(id: &str, text: &str) -> SearchDocument {
    SearchDocument {
        doc_id: DocumentId::new(id).unwrap(),
        path: NormalizedPath::new(format!("src/{id}.rs")).unwrap(),
        language: LanguageId::new("rust"),
        text: text.into(),
        embedding: None,
    }
}

fn send_request(
    stdin: &mut impl Write,
    stdout: &mut impl BufRead,
    request: &str,
) -> serde_json::Value {
    writeln!(stdin, "{request}").unwrap();
    stdin.flush().unwrap();
    let mut response = String::new();
    stdout.read_line(&mut response).unwrap();
    assert!(
        !response.is_empty(),
        "MCP server closed stdout unexpectedly"
    );
    serde_json::from_str(&response).unwrap()
}

#[tokio::test]
async fn idle_mcp_server_does_not_hold_the_duckdb_write_lock() {
    let directory = tempdir().unwrap();
    let database = directory.path().join("mcp.duckdb");
    let manifest = IndexManifest::lexical_v1();
    let index = DuckDbIndex::open(&database, manifest.clone(), None).unwrap();
    index
        .replace_all(&[document("before", "manifest validation")])
        .await
        .unwrap();
    drop(index);

    let child = Command::new(env!("CARGO_BIN_EXE_hay-mcp"))
        .args(["--backend", "duckdb", "--embeddings", "none", "--database"])
        .arg(&database)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut child = ChildGuard(child);
    let mut stdin = child.0.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.0.stdout.take().unwrap());

    let initialized = send_request(
        &mut stdin,
        &mut stdout,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"duckdb-lock-test","version":"1"}}}"#,
    );
    assert_eq!(initialized["id"], 1);
    stdin
        .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
        .unwrap();
    stdin.flush().unwrap();

    let updater = DuckDbIndex::open(&database, manifest.clone(), None).unwrap();
    updater
        .upsert_documents(&[document("after", "incremental update")])
        .await
        .unwrap();
    drop(updater);

    let capabilities = send_request(
        &mut stdin,
        &mut stdout,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"capabilities","arguments":{}}}"#,
    );
    assert_eq!(capabilities["id"], 2);
    assert!(
        capabilities.to_string().contains("\"document_count\":2"),
        "capabilities response did not observe the update: {capabilities}"
    );

    let updater = DuckDbIndex::open(&database, manifest, None).unwrap();
    updater
        .upsert_documents(&[document("final", "second update")])
        .await
        .unwrap();
}
