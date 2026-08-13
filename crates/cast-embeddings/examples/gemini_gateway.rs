use std::env;

use cast_embeddings::{CloudflareVertexGemini2, CloudflareVertexGemini2Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let token = env::var("COTH_HAY_SEEKER_CF_AIG_TOKEN").map_err(
        |_| "COTH_HAY_SEEKER_CF_AIG_TOKEN is missing; add the Cloudflare Gateway token to .env",
    )?;
    if token.trim().is_empty() {
        return Err(
            "COTH_HAY_SEEKER_CF_AIG_TOKEN is empty; add the Cloudflare Gateway token to .env"
                .into(),
        );
    }
    let endpoint = env::var("COTH_HAY_SEEKER_GEMINI_GATEWAY_URL").map_err(
        |_| "COTH_HAY_SEEKER_GEMINI_GATEWAY_URL is missing; add the complete gateway route to .env",
    )?;
    let mut config = CloudflareVertexGemini2Config::new(endpoint, token);
    if let Ok(dimensions) = env::var("COTH_HAY_SEEKER_GEMINI_EMBEDDING_DIMENSIONS") {
        config = config.with_dimensions(dimensions.parse()?);
    }
    if let Ok(concurrency) = env::var("COTH_HAY_SEEKER_GEMINI_EMBEDDING_CONCURRENCY") {
        config = config.with_max_concurrency(concurrency.parse()?);
    }

    let mut arguments = env::args().skip(1);
    let kind = arguments.next().unwrap_or_else(|| "query".into());
    let text = arguments.collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return Err("usage: gemini_gateway <query|document> <text>".into());
    }

    let embedder = CloudflareVertexGemini2::new(config)?;
    let vector = match kind.as_str() {
        "query" => embedder.embed_query(&text).await?,
        "document" => embedder.embed_document(&text).await?,
        _ => return Err("first argument must be query or document".into()),
    };
    let norm = vector
        .values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    println!(
        "provider={} model={} profile={} dimensions={} norm={norm:.6}",
        vector.identity.provider,
        vector.identity.model,
        vector.identity.profile,
        vector.values.len()
    );
    Ok(())
}
