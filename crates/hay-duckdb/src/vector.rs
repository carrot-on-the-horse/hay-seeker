use thiserror::Error;

const INT8_MAGIC: &[u8; 8] = b"HAYI8\x01\0\0";
const INT8_HEADER_BYTES: usize = 16;

#[derive(Debug, Eq, Error, PartialEq)]
pub(crate) enum VectorError {
    #[error("embedding must not be empty")]
    Empty,
    #[error("embedding contains a non-finite value")]
    NonFinite,
    #[error("stored embedding byte length is invalid")]
    InvalidByteLength,
    #[error("stored int8 embedding header is invalid")]
    InvalidHeader,
    #[error("stored embedding has {actual} dimensions; expected {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
}

pub(crate) struct PreparedQuery<'a> {
    values: &'a [f32],
    squared_norm: f32,
}

pub(crate) fn prepare_query(query: &[f32]) -> Result<PreparedQuery<'_>, VectorError> {
    validate_vector(query)?;
    Ok(PreparedQuery {
        values: query,
        squared_norm: query.iter().map(|value| value * value).sum(),
    })
}

pub(crate) fn encode_f32(vector: &[f32]) -> Result<Vec<u8>, VectorError> {
    validate_vector(vector)?;
    Ok(vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect())
}

pub(crate) fn decode_f32(blob: &[u8], dimensions: usize) -> Result<Vec<f32>, VectorError> {
    if blob.len() != dimensions.saturating_mul(size_of::<f32>()) {
        return Err(VectorError::InvalidByteLength);
    }
    let vector = blob
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    validate_vector(&vector)?;
    Ok(vector)
}

pub(crate) fn encode_int8(vector: &[f32]) -> Result<Vec<u8>, VectorError> {
    validate_vector(vector)?;
    let (minimum, maximum) = vector.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(*value), maximum.max(*value)),
    );
    let range = maximum - minimum;
    let (scale, offset) = if range <= f32::EPSILON {
        (0.0, minimum)
    } else {
        let scale = range / 255.0;
        (scale, minimum + 128.0 * scale)
    };

    let mut encoded = Vec::with_capacity(INT8_HEADER_BYTES + vector.len());
    encoded.extend_from_slice(INT8_MAGIC);
    encoded.extend_from_slice(&scale.to_le_bytes());
    encoded.extend_from_slice(&offset.to_le_bytes());
    encoded.extend(vector.iter().map(|value| {
        let quantized = if scale == 0.0 {
            0.0
        } else {
            ((*value - offset) / scale).round().clamp(-128.0, 127.0)
        };
        #[allow(clippy::cast_possible_truncation)]
        let quantized = quantized as i8;
        quantized.to_le_bytes()[0]
    }));
    Ok(encoded)
}

pub(crate) fn decode_int8(blob: &[u8], dimensions: usize) -> Result<Vec<f32>, VectorError> {
    let (scale, offset, values) = int8_parts(blob, dimensions)?;
    Ok(values
        .iter()
        .map(|value| f32::from(i8::from_le_bytes([*value])) * scale + offset)
        .collect())
}

pub(crate) fn cosine_f32(query: &PreparedQuery<'_>, right: &[f32]) -> Result<f32, VectorError> {
    if query.values.len() != right.len() {
        return Err(VectorError::DimensionMismatch {
            expected: query.values.len(),
            actual: right.len(),
        });
    }
    validate_vector(right)?;
    Ok(cosine_components(query, right.iter().copied()))
}

pub(crate) fn cosine_int8(query: &PreparedQuery<'_>, blob: &[u8]) -> Result<f32, VectorError> {
    let (scale, offset, values) = int8_parts(blob, query.values.len())?;
    let document = values
        .iter()
        .map(|value| f32::from(i8::from_le_bytes([*value])) * scale + offset);
    Ok(cosine_components(query, document))
}

fn int8_parts(blob: &[u8], dimensions: usize) -> Result<(f32, f32, &[u8]), VectorError> {
    if blob.len() != INT8_HEADER_BYTES.saturating_add(dimensions) {
        return Err(VectorError::InvalidByteLength);
    }
    if &blob[..INT8_MAGIC.len()] != INT8_MAGIC {
        return Err(VectorError::InvalidHeader);
    }
    let scale = f32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
    let offset = f32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
    if !scale.is_finite() || !offset.is_finite() || scale < 0.0 {
        return Err(VectorError::NonFinite);
    }
    Ok((scale, offset, &blob[INT8_HEADER_BYTES..]))
}

fn validate_vector(vector: &[f32]) -> Result<(), VectorError> {
    if vector.is_empty() {
        return Err(VectorError::Empty);
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(VectorError::NonFinite);
    }
    Ok(())
}

fn cosine_components(query: &PreparedQuery<'_>, right: impl Iterator<Item = f32>) -> f32 {
    let (dot, right_squared) =
        query.values.iter().copied().zip(right).fold(
            (0.0_f32, 0.0_f32),
            |(dot, right_squared), (left, right)| {
                (dot + left * right, right_squared + right * right)
            },
        );
    if query.squared_norm == 0.0 || right_squared == 0.0 {
        0.0
    } else {
        dot / (query.squared_norm.sqrt() * right_squared.sqrt())
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn int8_round_trip_is_bounded_and_versioned() {
        let vector = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let encoded = encode_int8(&vector).unwrap();
        assert_eq!(&encoded[..8], INT8_MAGIC);
        let decoded = decode_int8(&encoded, vector.len()).unwrap();
        for (expected, actual) in vector.iter().zip(decoded) {
            assert!((expected - actual).abs() <= 2.0 / 255.0);
        }
    }

    #[test]
    fn float_round_trip_remains_available_for_existing_hosted_profiles() {
        let vector = [0.25, -0.5, 1.0];
        let encoded = encode_f32(&vector).unwrap();
        assert_eq!(encoded.len(), vector.len() * size_of::<f32>());
        assert_eq!(decode_f32(&encoded, vector.len()).unwrap(), vector);
    }

    #[test]
    fn constant_vector_round_trip_does_not_divide_by_zero() {
        let encoded = encode_int8(&[0.25; 8]).unwrap();
        assert_eq!(decode_int8(&encoded, 8).unwrap(), vec![0.25; 8]);
    }

    #[test]
    fn quantized_cosine_preserves_nearest_direction() {
        let query = [0.8, 0.2, -0.1, 0.4];
        let query = prepare_query(&query).unwrap();
        let near = encode_int8(&[0.79, 0.21, -0.09, 0.41]).unwrap();
        let far = encode_int8(&[-0.8, -0.2, 0.1, -0.4]).unwrap();
        assert!(cosine_int8(&query, &near).unwrap() > cosine_int8(&query, &far).unwrap());
    }

    #[test]
    fn malformed_blob_is_rejected_before_scoring() {
        let query = prepare_query(&[1.0, 2.0]).unwrap();
        assert_eq!(
            cosine_int8(&query, &[0; 18]),
            Err(VectorError::InvalidHeader)
        );
    }
}
