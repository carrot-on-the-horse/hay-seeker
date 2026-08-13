use std::error::Error;

use cast_embeddings::{
    CloudflareWorkersAiEmbeddings, CloudflareWorkersAiEmbeddingsConfig, RetryPolicy,
    RetryingEmbedder,
};
use cast_index::{DocumentId, Embedder, EmbeddingInput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();
    let account_id = std::env::var("COTH_HAY_SEEKER_CLOUDFLARE_ACCOUNT_ID")?;
    let token = std::env::var("COTH_HAY_SEEKER_CLOUDFLARE_AI_TOKEN")?;
    let provider = CloudflareWorkersAiEmbeddings::new(CloudflareWorkersAiEmbeddingsConfig::new(
        account_id, token,
    ))?;
    let embedder = RetryingEmbedder::new(provider, RetryPolicy::default());
    let document_id = DocumentId::new("synthetic-route-registration")?;
    let vectors = embedder
        .embed_batch(&[EmbeddingInput {
            document_id: &document_id,
            text: "fn register_routes(router: &mut Router) { router.get(\"/health\", health); }",
        }])
        .await?;
    let query = embedder
        .embed_query("where are HTTP routes registered?")
        .await?;
    let document = vectors
        .first()
        .ok_or("Workers AI returned no document embedding")?;
    let similarity = cosine(&document.values, &query.values)?;
    println!(
        "provider={} model={} dimensions={} cosine={similarity:.6}",
        document.identity.provider,
        document.identity.model,
        document.values.len()
    );
    Ok(())
}

fn cosine(left: &[f32], right: &[f32]) -> Result<f32, Box<dyn Error>> {
    if left.len() != right.len() || left.is_empty() {
        return Err("embedding dimensions do not match".into());
    }
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return Err("embedding norm must not be zero".into());
    }
    Ok(dot / (left_norm * right_norm))
}
