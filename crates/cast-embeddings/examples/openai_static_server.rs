use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;

use cast_embeddings::{LocalStaticConfig, LocalStaticEmbedder};
use cast_index::{DocumentId, Embedder, EmbeddingInput};
use futures::executor::block_on;
use serde::{Deserialize, Serialize};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

#[derive(Deserialize)]
struct EmbeddingRequest {
    input: EmbeddingRequestInput,
    model: String,
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingRequestInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Serialize)]
struct EmbeddingResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: String,
    usage: EmbeddingUsage,
}

#[derive(Serialize)]
struct EmbeddingData {
    object: &'static str,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Serialize)]
struct EmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let bundle_dir = std::env::var("HAY_LOCAL_STATIC_MODEL_DIR")?;
    let address =
        std::env::var("HAY_OPENAI_STATIC_ADDRESS").unwrap_or_else(|_| "127.0.0.1:11435".into());
    let address: SocketAddr = address.parse()?;
    if !address.ip().is_loopback() {
        return Err("benchmark adapter must bind to a loopback address".into());
    }
    let embedder = Arc::new(LocalStaticEmbedder::new(LocalStaticConfig::new(
        bundle_dir,
    ))?);
    let listener = TcpListener::bind(address)?;
    eprintln!("local static OpenAI adapter listening on http://{address}/v1");

    for connection in listener.incoming() {
        match connection {
            Ok(stream) => {
                let embedder = Arc::clone(&embedder);
                std::thread::spawn(move || {
                    if let Err(error) = serve(stream, &embedder) {
                        eprintln!("local static OpenAI adapter request failed: {error}");
                    }
                });
            }
            Err(error) => eprintln!("local static OpenAI adapter accept failed: {error}"),
        }
    }
    Ok(())
}

fn serve(
    mut stream: TcpStream,
    embedder: &LocalStaticEmbedder,
) -> Result<(), Box<dyn std::error::Error>> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(30)))?;
    let request = read_request(&mut stream)?;
    let (status, content_type, body) = if request.method == "GET"
        && (request.path == "/v1/models" || request.path.starts_with("/v1/models/"))
    {
        (
            "200 OK",
            "application/json",
            serde_json::json!({
                "object": "list",
                "data": [{"id": embedder.identity().model, "object": "model"}]
            })
            .to_string(),
        )
    } else if request.method == "POST" && request.path == "/v1/embeddings" {
        match create_embeddings(embedder, &request.body) {
            Ok(body) => ("200 OK", "application/json", body),
            Err(error) => (
                "400 Bad Request",
                "application/json",
                serde_json::json!({"error": {"message": error.to_string(), "type": "invalid_request_error"}}).to_string(),
            ),
        }
    } else {
        (
            "404 Not Found",
            "application/json",
            serde_json::json!({"error": {"message": "not found", "type": "invalid_request_error"}})
                .to_string(),
        )
    };
    write_response(&mut stream, status, content_type, body.as_bytes())?;
    Ok(())
}

fn create_embeddings(
    embedder: &LocalStaticEmbedder,
    body: &[u8],
) -> Result<String, Box<dyn std::error::Error>> {
    let request: EmbeddingRequest = serde_json::from_slice(body)?;
    if request.model != embedder.identity().model
        && request.model != "potion-code-16m-v2"
        && request.model != "text-embedding-3-small"
    {
        return Err(format!("unsupported model {}", request.model).into());
    }
    if request.dimensions.is_some_and(|value| value != 256) {
        return Err("Potion code model supports exactly 256 dimensions".into());
    }
    let texts = match request.input {
        EmbeddingRequestInput::One(text) => vec![text],
        EmbeddingRequestInput::Many(texts) => texts,
    };
    if texts.is_empty() {
        return Err("embedding input must not be empty".into());
    }
    let ids = (0..texts.len())
        .map(|index| DocumentId::new(format!("openai-adapter-{index}")))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = ids
        .iter()
        .zip(&texts)
        .map(|(document_id, text)| EmbeddingInput { document_id, text })
        .collect::<Vec<_>>();
    let vectors = block_on(embedder.embed_batch(&inputs))?;
    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, vector)| EmbeddingData {
            object: "embedding",
            embedding: vector.values,
            index,
        })
        .collect();
    Ok(serde_json::to_string(&EmbeddingResponse {
        object: "list",
        data,
        model: request.model,
        usage: EmbeddingUsage {
            prompt_tokens: 0,
            total_tokens: 0,
        },
    })?)
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 8 * 1024];
    let header_end = loop {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err("connection closed before HTTP headers completed".into());
        }
        bytes.extend_from_slice(&buffer[..read]);
        if let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
        if bytes.len() > MAX_HEADER_BYTES {
            return Err("HTTP headers exceed the adapter limit".into());
        }
    };
    let headers = std::str::from_utf8(&bytes[..header_end])?;
    let request_line = headers
        .lines()
        .next()
        .ok_or("HTTP request line is missing")?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().ok_or("HTTP method is missing")?.into();
    let path = request_parts.next().ok_or("HTTP path is missing")?.into();
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(name, value)| {
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>())
            })
        })
        .transpose()?
        .unwrap_or(0);
    if content_length > MAX_BODY_BYTES {
        return Err("HTTP body exceeds the adapter limit".into());
    }
    let required = header_end
        .checked_add(content_length)
        .ok_or("HTTP request length overflow")?;
    while bytes.len() < required {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err("connection closed before HTTP body completed".into());
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    Ok(HttpRequest {
        method,
        path,
        body: bytes[header_end..required].to_vec(),
    })
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}
