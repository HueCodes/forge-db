//! Filter conditions for metadata-based filtering.

use super::value::MetadataValue;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// A condition for filtering vectors based on metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FilterCondition {
    /// Field equals value.
    Equals {
        field: String,
        value: MetadataValue,
    },

    /// Field does not equal value.
    NotEquals {
        field: String,
        value: MetadataValue,
    },

    /// Field is greater than value.
    GreaterThan {
        field: String,
        value: MetadataValue,
    },

    /// Field is greater than or equal to value.
    GreaterThanOrEqual {
        field: String,
        value: MetadataValue,
    },

    /// Field is less than value.
    LessThan {
        field: String,
        value: MetadataValue,
    },

    /// Field is less than or equal to value.
    LessThanOrEqual {
        field: String,
        value: MetadataValue,
    },

    /// Field value is in the given range [min, max].
    Range {
        field: String,
        min: MetadataValue,
        max: MetadataValue,
        /// Include min in range (default: true).
        min_inclusive: bool,
        /// Include max in range (default: true).
        max_inclusive: bool,
    },

    /// Field value is one of the given values.
    In {
        field: String,
        values: Vec<MetadataValue>,
    },

    /// Field value is not one of the given values.
    NotIn {
        field: String,
        values: Vec<MetadataValue>,
    },

    /// Field (array) contains the given value.
    Contains {
        field: String,
        value: MetadataValue,
    },

    /// Field exists and is not null.
    Exists { field: String },

    /// Field is null or does not exist.
    IsNull { field: String },

    /// String field starts with prefix.
    StartsWith { field: String, prefix: String },

    /// All conditions must be true.
    And(Vec<FilterCondition>),

    /// At least one condition must be true.
    Or(Vec<FilterCondition>),

    /// Condition must be false.
    Not(Box<FilterCondition>),
}

impl FilterCondition {
    /// Create an equality filter.
    pub fn eq(field: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        FilterCondition::Equals {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a not-equals filter.
    pub fn ne(field: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        FilterCondition::NotEquals {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a greater-than filter.
    pub fn gt(field: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        FilterCondition::GreaterThan {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a greater-than-or-equal filter.
    pub fn gte(field: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        FilterCondition::GreaterThanOrEqual {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a less-than filter.
    pub fn lt(field: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        FilterCondition::LessThan {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a less-than-or-equal filter.
    pub fn lte(field: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        FilterCondition::LessThanOrEqual {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a range filter (inclusive on both ends).
    pub fn range(
        field: impl Into<String>,
        min: impl Into<MetadataValue>,
        max: impl Into<MetadataValue>,
    ) -> Self {
        FilterCondition::Range {
            field: field.into(),
            min: min.into(),
            max: max.into(),
            min_inclusive: true,
            max_inclusive: true,
        }
    }

    /// Create an "in" filter.
    pub fn is_in(field: impl Into<String>, values: Vec<MetadataValue>) -> Self {
        FilterCondition::In {
            field: field.into(),
            values,
        }
    }

    /// Create a "not in" filter.
    pub fn not_in(field: impl Into<String>, values: Vec<MetadataValue>) -> Self {
        FilterCondition::NotIn {
            field: field.into(),
            values,
        }
    }

    /// Create an AND filter combining multiple conditions.
    pub fn and(conditions: Vec<FilterCondition>) -> Self {
        FilterCondition::And(conditions)
    }

    /// Create an OR filter combining multiple conditions.
    pub fn or(conditions: Vec<FilterCondition>) -> Self {
        FilterCondition::Or(conditions)
    }

    /// Create a NOT filter inverting a condition.
    pub fn negate(condition: FilterCondition) -> Self {
        FilterCondition::Not(Box::new(condition))
    }

    /// Evaluate this filter against a set of metadata.
    pub fn evaluate(&self, metadata: &std::collections::HashMap<String, MetadataValue>) -> bool {
        match self {
            FilterCondition::Equals { field, value } => {
                metadata.get(field).map(|v| v == value).unwrap_or(false)
            }

            FilterCondition::NotEquals { field, value } => {
                metadata.get(field).map(|v| v != value).unwrap_or(true)
            }

            FilterCondition::GreaterThan { field, value } => metadata
                .get(field)
                .and_then(|v| v.partial_cmp_value(value))
                .map(|ord| ord == Ordering::Greater)
                .unwrap_or(false),

            FilterCondition::GreaterThanOrEqual { field, value } => metadata
                .get(field)
                .and_then(|v| v.partial_cmp_value(value))
                .map(|ord| ord != Ordering::Less)
                .unwrap_or(false),

            FilterCondition::LessThan { field, value } => metadata
                .get(field)
                .and_then(|v| v.partial_cmp_value(value))
                .map(|ord| ord == Ordering::Less)
                .unwrap_or(false),

            FilterCondition::LessThanOrEqual { field, value } => metadata
                .get(field)
                .and_then(|v| v.partial_cmp_value(value))
                .map(|ord| ord != Ordering::Greater)
                .unwrap_or(false),

            FilterCondition::Range {
                field,
                min,
                max,
                min_inclusive,
                max_inclusive,
            } => {
                metadata
                    .get(field)
                    .map(|v| {
                        let min_ok = match v.partial_cmp_value(min) {
                            Some(Ordering::Greater) => true,
                            Some(Ordering::Equal) => *min_inclusive,
                            _ => false,
                        };
                        let max_ok = match v.partial_cmp_value(max) {
                            Some(Ordering::Less) => true,
                            Some(Ordering::Equal) => *max_inclusive,
                            _ => false,
                        };
                        min_ok && max_ok
                    })
                    .unwrap_or(false)
            }

            FilterCondition::In { field, values } => metadata
                .get(field)
                .map(|v| values.contains(v))
                .unwrap_or(false),

            FilterCondition::NotIn { field, values } => metadata
                .get(field)
                .map(|v| !values.contains(v))
                .unwrap_or(true),

            FilterCondition::Contains { field, value } => metadata
                .get(field)
                .map(|v| value.is_contained_in(v))
                .unwrap_or(false),

            FilterCondition::Exists { field } => {
                metadata.get(field).map(|v| !v.is_null()).unwrap_or(false)
            }

            FilterCondition::IsNull { field } => {
                metadata.get(field).map(|v| v.is_null()).unwrap_or(true)
            }

            FilterCondition::StartsWith { field, prefix } => metadata
                .get(field)
                .and_then(|v| v.as_str())
                .map(|s| s.starts_with(prefix))
                .unwrap_or(false),

            FilterCondition::And(conditions) => conditions.iter().all(|c| c.evaluate(metadata)),

            FilterCondition::Or(conditions) => conditions.iter().any(|c| c.evaluate(metadata)),

            FilterCondition::Not(condition) => !condition.evaluate(metadata),
        }
    }

    /// Get the fields referenced by this filter condition.
    pub fn referenced_fields(&self) -> Vec<&str> {
        match self {
            FilterCondition::Equals { field, .. }
            | FilterCondition::NotEquals { field, .. }
            | FilterCondition::GreaterThan { field, .. }
            | FilterCondition::GreaterThanOrEqual { field, .. }
            | FilterCondition::LessThan { field, .. }
            | FilterCondition::LessThanOrEqual { field, .. }
            | FilterCondition::Range { field, .. }
            | FilterCondition::In { field, .. }
            | FilterCondition::NotIn { field, .. }
            | FilterCondition::Contains { field, .. }
            | FilterCondition::Exists { field }
            | FilterCondition::IsNull { field }
            | FilterCondition::StartsWith { field, .. } => vec![field.as_str()],

            FilterCondition::And(conditions) | FilterCondition::Or(conditions) => {
                conditions.iter().flat_map(|c| c.referenced_fields()).collect()
            }

            FilterCondition::Not(condition) => condition.referenced_fields(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_metadata() -> HashMap<String, MetadataValue> {
        let mut m = HashMap::new();
        m.insert("name".into(), MetadataValue::String("Alice".into()));
        m.insert("age".into(), MetadataValue::Integer(30));
        m.insert("score".into(), MetadataValue::Float(85.5));
        m.insert("active".into(), MetadataValue::Boolean(true));
        m.insert(
            "tags".into(),
            MetadataValue::Array(vec![
                MetadataValue::String("rust".into()),
                MetadataValue::String("python".into()),
            ]),
        );
        m
    }

    #[test]
    fn test_equals() {
        let m = make_metadata();
        let filter = FilterCondition::eq("name", "Alice");
        assert!(filter.evaluate(&m));

        let filter = FilterCondition::eq("name", "Bob");
        assert!(!filter.evaluate(&m));
    }

    #[test]
    fn test_not_equals() {
        let m = make_metadata();
        let filter = FilterCondition::ne("name", "Bob");
        assert!(filter.evaluate(&m));
    }

    #[test]
    fn test_comparison() {
        let m = make_metadata();

        assert!(FilterCondition::gt("age", 25i64).evaluate(&m));
        assert!(!FilterCondition::gt("age", 35i64).evaluate(&m));

        assert!(FilterCondition::gte("age", 30i64).evaluate(&m));
        assert!(FilterCondition::lte("age", 30i64).evaluate(&m));

        assert!(FilterCondition::lt("age", 35i64).evaluate(&m));
    }

    #[test]
    fn test_range() {
        let m = make_metadata();

        let filter = FilterCondition::range("age", 25i64, 35i64);
        assert!(filter.evaluate(&m));

        let filter = FilterCondition::range("age", 31i64, 40i64);
        assert!(!filter.evaluate(&m));
    }

    #[test]
    fn test_in() {
        let m = make_metadata();

        let filter = FilterCondition::is_in(
            "name",
            vec![
                MetadataValue::String("Alice".into()),
                MetadataValue::String("Bob".into()),
            ],
        );
        assert!(filter.evaluate(&m));

        let filter = FilterCondition::is_in(
            "name",
            vec![
                MetadataValue::String("Bob".into()),
                MetadataValue::String("Carol".into()),
            ],
        );
        assert!(!filter.evaluate(&m));
    }

    #[test]
    fn test_contains() {
        let m = make_metadata();

        let filter = FilterCondition::Contains {
            field: "tags".into(),
            value: MetadataValue::String("rust".into()),
        };
        assert!(filter.evaluate(&m));

        let filter = FilterCondition::Contains {
            field: "tags".into(),
            value: MetadataValue::String("java".into()),
        };
        assert!(!filter.evaluate(&m));
    }

    #[test]
    fn test_exists() {
        let m = make_metadata();

        let filter = FilterCondition::Exists {
            field: "name".into(),
        };
        assert!(filter.evaluate(&m));

        let filter = FilterCondition::Exists {
            field: "missing".into(),
        };
        assert!(!filter.evaluate(&m));
    }

    #[test]
    fn test_and_or() {
        let m = make_metadata();

        let filter = FilterCondition::and(vec![
            FilterCondition::eq("name", "Alice"),
            FilterCondition::gt("age", 25i64),
        ]);
        assert!(filter.evaluate(&m));

        let filter = FilterCondition::and(vec![
            FilterCondition::eq("name", "Alice"),
            FilterCondition::gt("age", 35i64),
        ]);
        assert!(!filter.evaluate(&m));

        let filter = FilterCondition::or(vec![
            FilterCondition::eq("name", "Bob"),
            FilterCondition::gt("age", 25i64),
        ]);
        assert!(filter.evaluate(&m));
    }

    #[test]
    fn test_not() {
        let m = make_metadata();

        let filter = FilterCondition::negate(FilterCondition::eq("name", "Bob"));
        assert!(filter.evaluate(&m));
    }

    #[test]
    fn test_starts_with() {
        let m = make_metadata();

        let filter = FilterCondition::StartsWith {
            field: "name".into(),
            prefix: "Ali".into(),
        };
        assert!(filter.evaluate(&m));

        let filter = FilterCondition::StartsWith {
            field: "name".into(),
            prefix: "Bob".into(),
        };
        assert!(!filter.evaluate(&m));
    }
}
