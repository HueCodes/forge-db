//! Metadata value types.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A metadata value that can be attached to a vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    /// String value.
    String(String),
    /// 64-bit signed integer.
    Integer(i64),
    /// 64-bit floating point number.
    Float(f64),
    /// Boolean value.
    Boolean(bool),
    /// Array of values (homogeneous types recommended).
    Array(Vec<MetadataValue>),
    /// Null/missing value.
    Null,
}

impl Eq for MetadataValue {}

impl Hash for MetadataValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash discriminant first to distinguish types
        std::mem::discriminant(self).hash(state);

        match self {
            MetadataValue::String(s) => s.hash(state),
            MetadataValue::Integer(i) => i.hash(state),
            MetadataValue::Float(f) => {
                // Use bit representation for consistent hashing
                // NaN values will all hash the same, which is fine
                f.to_bits().hash(state);
            }
            MetadataValue::Boolean(b) => b.hash(state),
            MetadataValue::Array(arr) => {
                arr.len().hash(state);
                for item in arr {
                    item.hash(state);
                }
            }
            MetadataValue::Null => {}
        }
    }
}

impl MetadataValue {
    /// Get the type name as a string (for error messages).
    pub fn type_name(&self) -> &'static str {
        match self {
            MetadataValue::String(_) => "string",
            MetadataValue::Integer(_) => "integer",
            MetadataValue::Float(_) => "float",
            MetadataValue::Boolean(_) => "boolean",
            MetadataValue::Array(_) => "array",
            MetadataValue::Null => "null",
        }
    }

    /// Check if this value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, MetadataValue::Null)
    }

    /// Try to get as string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as integer.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            MetadataValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to get as float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            MetadataValue::Float(f) => Some(*f),
            MetadataValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to get as boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as array reference.
    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            MetadataValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Compare two values for ordering (used in range queries).
    /// Returns None if types are incompatible.
    pub fn partial_cmp_value(&self, other: &MetadataValue) -> Option<Ordering> {
        match (self, other) {
            (MetadataValue::Integer(a), MetadataValue::Integer(b)) => a.partial_cmp(b),
            (MetadataValue::Float(a), MetadataValue::Float(b)) => a.partial_cmp(b),
            (MetadataValue::Integer(a), MetadataValue::Float(b)) => (*a as f64).partial_cmp(b),
            (MetadataValue::Float(a), MetadataValue::Integer(b)) => a.partial_cmp(&(*b as f64)),
            (MetadataValue::String(a), MetadataValue::String(b)) => a.partial_cmp(b),
            (MetadataValue::Boolean(a), MetadataValue::Boolean(b)) => a.partial_cmp(b),
            _ => None,
        }
    }

    /// Check if this value is contained in an array value.
    pub fn is_contained_in(&self, array: &MetadataValue) -> bool {
        match array {
            MetadataValue::Array(arr) => arr.contains(self),
            _ => false,
        }
    }
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        MetadataValue::String(s)
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        MetadataValue::String(s.to_string())
    }
}

impl From<i64> for MetadataValue {
    fn from(i: i64) -> Self {
        MetadataValue::Integer(i)
    }
}

impl From<i32> for MetadataValue {
    fn from(i: i32) -> Self {
        MetadataValue::Integer(i as i64)
    }
}

impl From<f64> for MetadataValue {
    fn from(f: f64) -> Self {
        MetadataValue::Float(f)
    }
}

impl From<f32> for MetadataValue {
    fn from(f: f32) -> Self {
        MetadataValue::Float(f as f64)
    }
}

impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        MetadataValue::Boolean(b)
    }
}

impl<T: Into<MetadataValue>> From<Vec<T>> for MetadataValue {
    fn from(v: Vec<T>) -> Self {
        MetadataValue::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<MetadataValue>> From<Option<T>> for MetadataValue {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => v.into(),
            None => MetadataValue::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_name() {
        assert_eq!(MetadataValue::String("test".into()).type_name(), "string");
        assert_eq!(MetadataValue::Integer(42).type_name(), "integer");
        assert_eq!(MetadataValue::Float(3.14).type_name(), "float");
        assert_eq!(MetadataValue::Boolean(true).type_name(), "boolean");
        assert_eq!(MetadataValue::Null.type_name(), "null");
    }

    #[test]
    fn test_conversions() {
        let s: MetadataValue = "hello".into();
        assert_eq!(s.as_str(), Some("hello"));

        let i: MetadataValue = 42i64.into();
        assert_eq!(i.as_integer(), Some(42));

        let f: MetadataValue = 3.14f64.into();
        assert_eq!(f.as_float(), Some(3.14));

        let b: MetadataValue = true.into();
        assert_eq!(b.as_bool(), Some(true));
    }

    #[test]
    fn test_partial_cmp() {
        let a = MetadataValue::Integer(10);
        let b = MetadataValue::Integer(20);
        assert_eq!(a.partial_cmp_value(&b), Some(Ordering::Less));

        let c = MetadataValue::Float(10.0);
        assert_eq!(a.partial_cmp_value(&c), Some(Ordering::Equal));
    }

    #[test]
    fn test_is_contained_in() {
        let value = MetadataValue::String("apple".into());
        let array = MetadataValue::Array(vec![
            MetadataValue::String("apple".into()),
            MetadataValue::String("banana".into()),
        ]);

        assert!(value.is_contained_in(&array));

        let other = MetadataValue::String("cherry".into());
        assert!(!other.is_contained_in(&array));
    }
}
