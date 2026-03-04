//! API key authentication middleware for forge-server.

use constant_time_eq::constant_time_eq;
use sha2::{Digest, Sha256};

/// Compute the SHA-256 hex digest of a raw API key.
pub fn hash_api_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hex::encode(hasher.finalize())
}

/// Verify a raw API key against a list of stored hashes in constant time.
///
/// Returns `true` if `key` matches any entry in `hashes`.
pub fn verify_api_key(key: &str, hashes: &[String]) -> bool {
    let key_hash = hash_api_key(key);
    let key_bytes = key_hash.as_bytes();

    hashes.iter().any(|stored| {
        let stored_bytes = stored.as_bytes();
        // Use constant-time comparison to prevent timing attacks.
        // If lengths differ, compare with a dummy to prevent early exit.
        if stored_bytes.len() != key_bytes.len() {
            // Still run the comparison to maintain constant time.
            let dummy = vec![0u8; stored_bytes.len()];
            let _ = constant_time_eq(stored_bytes, &dummy);
            false
        } else {
            constant_time_eq(stored_bytes, key_bytes)
        }
    })
}

/// Extract the bearer token from an `Authorization: Bearer <token>` header.
pub fn extract_bearer_token(header_value: &str) -> Option<&str> {
    header_value.strip_prefix("Bearer ").map(str::trim)
}

/// Extract the API key from an `X-API-Key: <key>` header or bearer token.
pub fn extract_api_key(
    authorization: Option<&str>,
    x_api_key: Option<&str>,
) -> Option<String> {
    // Prefer X-API-Key header, fall back to Authorization: Bearer
    if let Some(key) = x_api_key {
        return Some(key.trim().to_string());
    }
    if let Some(auth) = authorization {
        if let Some(token) = extract_bearer_token(auth) {
            return Some(token.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_roundtrip() {
        let key = "my-secret-api-key";
        let hash = hash_api_key(key);
        assert_eq!(hash.len(), 64); // SHA-256 hex = 64 chars
        assert!(verify_api_key(key, &[hash]));
    }

    #[test]
    fn test_wrong_key_rejected() {
        let hash = hash_api_key("correct-key");
        assert!(!verify_api_key("wrong-key", &[hash]));
    }

    #[test]
    fn test_empty_hashes() {
        assert!(!verify_api_key("any-key", &[]));
    }

    #[test]
    fn test_extract_bearer_token() {
        assert_eq!(extract_bearer_token("Bearer my-token"), Some("my-token"));
        assert_eq!(extract_bearer_token("Basic xyz"), None);
    }

    #[test]
    fn test_extract_api_key_prefers_x_api_key() {
        let result = extract_api_key(
            Some("Bearer bearer-token"),
            Some("x-api-key-value"),
        );
        assert_eq!(result.as_deref(), Some("x-api-key-value"));
    }

    #[test]
    fn test_extract_api_key_falls_back_to_bearer() {
        let result = extract_api_key(Some("Bearer bearer-token"), None);
        assert_eq!(result.as_deref(), Some("bearer-token"));
    }
}
