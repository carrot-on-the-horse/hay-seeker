#![forbid(unsafe_code)]

use std::num::NonZeroUsize;
use std::sync::Arc;

use cast_core::LanguageId;
use cast_index::{
    BoxFuture, ContentHash, DocumentId, Embedder, EmbeddingIdentity, EmbeddingInput,
    EmbeddingVector, HashAlgorithm, IndexError, NormalizedPath,
};
use hay_elasticsearch::{ElasticsearchConfig, ElasticsearchIndex};
use hay_search::{
    FdeParams, IndexManifest, Quantization, Query, Retriever, SearchDocument, SearchOpts,
};

struct TestEmbedder {
    identity: EmbeddingIdentity,
}

impl TestEmbedder {
    fn new() -> Self {
        Self {
            identity: EmbeddingIdentity {
                provider: "live-test".into(),
                model: "live-test-embedding".into(),
                dimensions: 64,
                profile: "live-test-asymmetric-v1".into(),
            },
        }
    }

    fn vector(&self, text: &str) -> EmbeddingVector {
        let mut values = vec![0.0; self.identity.dimensions];
        values[usize::from(!text.contains("route"))] = 1.0;
        EmbeddingVector {
            identity: self.identity.clone(),
            values,
        }
    }
}

impl Embedder for TestEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    fn embed_batch<'a>(
        &'a self,
        inputs: &'a [EmbeddingInput<'a>],
    ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
        Box::pin(async move { Ok(inputs.iter().map(|input| self.vector(input.text)).collect()) })
    }

    fn embed_query<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
        Box::pin(async move { Ok(self.vector(text)) })
    }
}

/// Exercises create, bulk, alias swap, manifest validation, search, and mget
/// against an explicitly supplied disposable Elasticsearch alias.
#[tokio::test]
#[ignore = "requires ELASTICSEARCH_TEST_URL and a disposable ELASTICSEARCH_TEST_INDEX"]
#[allow(clippy::too_many_lines)]
async fn live_elasticsearch_full_cycle() {
    let endpoint = std::env::var("ELASTICSEARCH_TEST_URL").expect("ELASTICSEARCH_TEST_URL");
    let alias =
        std::env::var("ELASTICSEARCH_TEST_INDEX").expect("ELASTICSEARCH_TEST_INDEX is disposable");
    let mut config = ElasticsearchConfig::new(&endpoint, &alias);
    if let Ok(api_key) = std::env::var("ELASTICSEARCH_API_KEY") {
        config = config.with_api_key(api_key);
    } else if let Ok(token) = std::env::var("ELASTICSEARCH_BEARER_TOKEN") {
        config = config.with_bearer_token(token);
    }
    let index = ElasticsearchIndex::new(config, IndexManifest::lexical_v1(), None).unwrap();
    let document_id = DocumentId::new("live-manifest-contract").unwrap();
    index
        .replace_all(&[SearchDocument {
            doc_id: document_id.clone(),
            path: NormalizedPath::new("docs/manifest-contract.md").unwrap(),
            language: LanguageId::new("markdown"),
            text: "runtime validates every persisted index manifest field".into(),
            embedding: None,
        }])
        .await
        .unwrap();
    assert!((1..=2).contains(&index.physical_generation_count().await.unwrap()));

    let results = index
        .search(
            &Query::new("persisted manifest validation").unwrap(),
            &SearchOpts {
                top_k: NonZeroUsize::new(1).unwrap(),
                candidate_limit: NonZeroUsize::new(10).unwrap(),
                enable_late_interaction: false,
            },
        )
        .await
        .unwrap();
    assert_eq!(results[0].doc_id, document_id);
    assert_eq!(index.document_count().await.unwrap(), 1);
    assert_eq!(
        index
            .documents(std::slice::from_ref(&document_id))
            .await
            .unwrap()
            .len(),
        1
    );

    let replacement_id = DocumentId::new("live-incremental-replacement").unwrap();
    index
        .update_stream(
            [Ok(SearchDocument {
                doc_id: replacement_id.clone(),
                path: NormalizedPath::new("src/incremental.rs").unwrap(),
                language: LanguageId::new("rust"),
                text: "incremental generation replaces obsolete chunks".into(),
                embedding: None,
            })],
            || Ok(vec![document_id.clone()]),
        )
        .await
        .unwrap();
    assert!((1..=2).contains(&index.physical_generation_count().await.unwrap()));
    assert_eq!(index.document_count().await.unwrap(), 1);
    assert!(index.documents(&[document_id]).await.unwrap().is_empty());
    assert_eq!(
        index
            .documents(std::slice::from_ref(&replacement_id))
            .await
            .unwrap()
            .len(),
        1
    );

    let error = index
        .update_stream(
            [Ok(SearchDocument {
                doc_id: DocumentId::new("never-published").unwrap(),
                path: NormalizedPath::new("src/never.rs").unwrap(),
                language: LanguageId::new("rust"),
                text: "this generation must not publish".into(),
                embedding: None,
            })],
            || Err(hay_search::SearchError::Corpus("checkpoint failure".into())),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("checkpoint failure"));
    assert_eq!(index.document_count().await.unwrap(), 1);
    assert!((1..=2).contains(&index.physical_generation_count().await.unwrap()));
    assert_eq!(
        index
            .documents(std::slice::from_ref(&replacement_id))
            .await
            .unwrap()
            .len(),
        1
    );

    let error = index
        .replace_stream(std::iter::once(Err(hay_search::SearchError::Corpus(
            "simulated source failure".into(),
        ))))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("simulated source failure"));
    assert_eq!(index.document_count().await.unwrap(), 1);
    assert!((1..=2).contains(&index.physical_generation_count().await.unwrap()));

    let mut dense_config = ElasticsearchConfig::new(&endpoint, format!("{alias}-dense"));
    if let Ok(api_key) = std::env::var("ELASTICSEARCH_API_KEY") {
        dense_config = dense_config.with_api_key(api_key);
    } else if let Ok(token) = std::env::var("ELASTICSEARCH_BEARER_TOKEN") {
        dense_config = dense_config.with_bearer_token(token);
    }
    let embedder: Arc<dyn Embedder> = Arc::new(TestEmbedder::new());
    let dense_manifest = IndexManifest {
        model_id: embedder.identity().model.clone(),
        model_revision: "live-test-revision".into(),
        embedding_profile: embedder.identity().profile.clone(),
        embed_dim: 64,
        mrl_dim: 64,
        quantization: Quantization::ElasticBbq,
        tokenizer_hash: ContentHash::new(HashAlgorithm::Sha256, "a".repeat(64)).unwrap(),
        chunker_version: "live-test-chunker".into(),
        fde_params: FdeParams::Disabled,
        schema_version: 1,
    };
    let dense = ElasticsearchIndex::new(dense_config, dense_manifest, Some(embedder)).unwrap();
    dense
        .replace_all(&[
            SearchDocument {
                doc_id: DocumentId::new("routes").unwrap(),
                path: NormalizedPath::new("src/routes.rs").unwrap(),
                language: LanguageId::new("rust"),
                text: "register health route and search route".into(),
                embedding: None,
            },
            SearchDocument {
                doc_id: DocumentId::new("manifest").unwrap(),
                path: NormalizedPath::new("src/manifest.rs").unwrap(),
                language: LanguageId::new("rust"),
                text: "validate stored index metadata".into(),
                embedding: None,
            },
        ])
        .await
        .unwrap();
    assert!((1..=2).contains(&dense.physical_generation_count().await.unwrap()));
    let results = dense
        .search(
            &Query::new("where is the route").unwrap(),
            &SearchOpts::default(),
        )
        .await
        .unwrap();
    assert_eq!(results[0].doc_id.as_str(), "routes");
    assert!(results[0].signals.lexical.is_some());
    assert!(results[0].signals.dense.is_some());
}
