//! Metadata storage with bitmap indices.

use super::filter::FilterCondition;
use super::value::MetadataValue;
use roaring::RoaringBitmap;
use std::collections::HashMap;

/// Storage for vector metadata with optional bitmap indices.
///
/// The store maintains metadata for vectors identified by u64 IDs and can
/// build bitmap indices for efficient filtering on specific fields.
pub struct MetadataStore {
    /// Metadata for each vector: id -> (field -> value)
    data: HashMap<u64, HashMap<String, MetadataValue>>,

    /// Bitmap indices for indexed fields.
    /// Structure: field_name -> (value_hash -> bitmap of matching IDs)
    /// Only stores exact-match indices for string and integer fields.
    indices: HashMap<String, HashMap<u64, RoaringBitmap>>,

    /// Fields that have been indexed.
    indexed_fields: Vec<String>,

    /// ID to internal index mapping (for bitmap operations).
    /// Bitmaps use u32 indices, so we map u64 vector IDs to u32 positions.
    id_to_index: HashMap<u64, u32>,
    index_to_id: Vec<u64>,
}

impl Default for MetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataStore {
    /// Create a new empty metadata store.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            indices: HashMap::new(),
            indexed_fields: Vec::new(),
            id_to_index: HashMap::new(),
            index_to_id: Vec::new(),
        }
    }

    /// Create a store with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
            indices: HashMap::new(),
            indexed_fields: Vec::new(),
            id_to_index: HashMap::with_capacity(capacity),
            index_to_id: Vec::with_capacity(capacity),
        }
    }

    /// Get the number of vectors with metadata.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Insert or update metadata for a vector.
    pub fn insert(&mut self, vector_id: u64, field: impl Into<String>, value: MetadataValue) {
        // Ensure ID mapping exists
        if let std::collections::hash_map::Entry::Vacant(e) = self.id_to_index.entry(vector_id) {
            let idx = self.index_to_id.len() as u32;
            e.insert(idx);
            self.index_to_id.push(vector_id);
        }

        let entry = self.data.entry(vector_id).or_default();
        entry.insert(field.into(), value);
    }

    /// Insert all metadata for a vector at once.
    pub fn insert_all(&mut self, vector_id: u64, metadata: HashMap<String, MetadataValue>) {
        // Ensure ID mapping exists
        if let std::collections::hash_map::Entry::Vacant(e) = self.id_to_index.entry(vector_id) {
            let idx = self.index_to_id.len() as u32;
            e.insert(idx);
            self.index_to_id.push(vector_id);
        }

        self.data.insert(vector_id, metadata);
    }

    /// Get metadata for a vector.
    pub fn get(&self, vector_id: u64) -> Option<&HashMap<String, MetadataValue>> {
        self.data.get(&vector_id)
    }

    /// Get a specific field value for a vector.
    pub fn get_field(&self, vector_id: u64, field: &str) -> Option<&MetadataValue> {
        self.data.get(&vector_id).and_then(|m| m.get(field))
    }

    /// Remove metadata for a vector.
    pub fn remove(&mut self, vector_id: u64) -> Option<HashMap<String, MetadataValue>> {
        // Note: We don't remove from id_to_index/index_to_id to avoid invalidating bitmaps.
        // The indices will need to be rebuilt after removals.
        self.data.remove(&vector_id)
    }

    /// Build or rebuild the bitmap index for a field.
    ///
    /// This creates a bitmap index that maps field values to the set of vector IDs
    /// that have that value. Only works well for low-cardinality fields (< 10,000 unique values).
    pub fn build_index(&mut self, field: &str) {
        let mut value_map: HashMap<u64, RoaringBitmap> = HashMap::new();

        for (&vector_id, metadata) in &self.data {
            if let Some(value) = metadata.get(field) {
                let value_hash = hash_value(value);
                let bitmap_idx = self.id_to_index[&vector_id];

                value_map
                    .entry(value_hash)
                    .or_default()
                    .insert(bitmap_idx);
            }
        }

        self.indices.insert(field.to_string(), value_map);

        if !self.indexed_fields.contains(&field.to_string()) {
            self.indexed_fields.push(field.to_string());
        }
    }

    /// Check if a field has been indexed.
    pub fn is_indexed(&self, field: &str) -> bool {
        self.indices.contains_key(field)
    }

    /// Get the list of indexed fields.
    pub fn indexed_fields(&self) -> &[String] {
        &self.indexed_fields
    }

    /// Filter vectors using a condition, returning matching vector IDs.
    ///
    /// If the filter can use bitmap indices, it will do so for efficiency.
    /// Otherwise, it falls back to scanning all metadata.
    pub fn filter(&self, condition: &FilterCondition) -> Vec<u64> {
        // Try to use bitmap index first
        if let Some(bitmap) = self.filter_with_index(condition) {
            return bitmap
                .iter()
                .map(|idx| self.index_to_id[idx as usize])
                .collect();
        }

        // Fall back to scanning
        self.data
            .iter()
            .filter(|(_, metadata)| condition.evaluate(metadata))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Filter using bitmap index if possible.
    fn filter_with_index(&self, condition: &FilterCondition) -> Option<RoaringBitmap> {
        match condition {
            FilterCondition::Equals { field, value } => {
                if let Some(field_index) = self.indices.get(field) {
                    let value_hash = hash_value(value);
                    return Some(field_index.get(&value_hash).cloned().unwrap_or_default());
                }
                None
            }

            FilterCondition::In { field, values } => {
                if let Some(field_index) = self.indices.get(field) {
                    let mut result = RoaringBitmap::new();
                    for value in values {
                        let value_hash = hash_value(value);
                        if let Some(bitmap) = field_index.get(&value_hash) {
                            result |= bitmap;
                        }
                    }
                    return Some(result);
                }
                None
            }

            FilterCondition::And(conditions) => {
                let mut result: Option<RoaringBitmap> = None;

                for cond in conditions {
                    if let Some(bitmap) = self.filter_with_index(cond) {
                        result = Some(match result {
                            Some(r) => r & bitmap,
                            None => bitmap,
                        });
                    }
                }

                result
            }

            FilterCondition::Or(conditions) => {
                let mut result = RoaringBitmap::new();
                let mut all_indexed = true;

                for cond in conditions {
                    if let Some(bitmap) = self.filter_with_index(cond) {
                        result |= bitmap;
                    } else {
                        all_indexed = false;
                        break;
                    }
                }

                if all_indexed {
                    Some(result)
                } else {
                    None
                }
            }

            FilterCondition::Not(inner) => {
                if let Some(bitmap) = self.filter_with_index(inner) {
                    // Create universe bitmap (all IDs)
                    let mut universe = RoaringBitmap::new();
                    for idx in 0..self.index_to_id.len() as u32 {
                        universe.insert(idx);
                    }
                    return Some(universe - bitmap);
                }
                None
            }

            // Other conditions don't use bitmap indices
            _ => None,
        }
    }

    /// Get matching vector IDs as a bitmap (for integration with search).
    pub fn filter_bitmap(&self, condition: &FilterCondition) -> RoaringBitmap {
        if let Some(bitmap) = self.filter_with_index(condition) {
            bitmap
        } else {
            // Fall back to scanning and building bitmap
            let mut bitmap = RoaringBitmap::new();
            for (&vector_id, metadata) in &self.data {
                if condition.evaluate(metadata) {
                    if let Some(&idx) = self.id_to_index.get(&vector_id) {
                        bitmap.insert(idx);
                    }
                }
            }
            bitmap
        }
    }

    /// Check if a vector ID matches a filter condition.
    pub fn matches(&self, vector_id: u64, condition: &FilterCondition) -> bool {
        self.data
            .get(&vector_id)
            .map(|m| condition.evaluate(m))
            .unwrap_or(false)
    }

    /// Get internal index for a vector ID (for bitmap operations).
    pub fn get_index(&self, vector_id: u64) -> Option<u32> {
        self.id_to_index.get(&vector_id).copied()
    }

    /// Get vector ID from internal index.
    pub fn get_id(&self, index: u32) -> Option<u64> {
        self.index_to_id.get(index as usize).copied()
    }
}

/// Hash a metadata value for indexing.
fn hash_value(value: &MetadataValue) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    match value {
        MetadataValue::String(s) => {
            0u8.hash(&mut hasher);
            s.hash(&mut hasher);
        }
        MetadataValue::Integer(i) => {
            1u8.hash(&mut hasher);
            i.hash(&mut hasher);
        }
        MetadataValue::Float(f) => {
            2u8.hash(&mut hasher);
            f.to_bits().hash(&mut hasher);
        }
        MetadataValue::Boolean(b) => {
            3u8.hash(&mut hasher);
            b.hash(&mut hasher);
        }
        MetadataValue::Null => {
            4u8.hash(&mut hasher);
        }
        MetadataValue::Array(arr) => {
            5u8.hash(&mut hasher);
            for item in arr {
                hash_value(item).hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_store() -> MetadataStore {
        let mut store = MetadataStore::new();

        // Add test data
        for i in 0..100u64 {
            let category = if i % 3 == 0 {
                "electronics"
            } else if i % 3 == 1 {
                "clothing"
            } else {
                "books"
            };

            store.insert(i, "category", MetadataValue::String(category.into()));
            store.insert(i, "price", MetadataValue::Float((i as f64) * 10.0));
            store.insert(i, "in_stock", MetadataValue::Boolean(i % 2 == 0));
        }

        store
    }

    #[test]
    fn test_insert_and_get() {
        let mut store = MetadataStore::new();
        store.insert(1, "name", MetadataValue::String("test".into()));
        store.insert(1, "count", MetadataValue::Integer(42));

        assert_eq!(store.len(), 1);
        assert_eq!(
            store.get_field(1, "name"),
            Some(&MetadataValue::String("test".into()))
        );
        assert_eq!(
            store.get_field(1, "count"),
            Some(&MetadataValue::Integer(42))
        );
    }

    #[test]
    fn test_filter_without_index() {
        let store = create_test_store();

        let filter = FilterCondition::eq("category", "electronics");
        let results = store.filter(&filter);

        assert_eq!(results.len(), 34); // 0, 3, 6, ... up to 99
        for id in &results {
            assert_eq!(id % 3, 0);
        }
    }

    #[test]
    fn test_filter_with_index() {
        let mut store = create_test_store();
        store.build_index("category");

        assert!(store.is_indexed("category"));

        let filter = FilterCondition::eq("category", "electronics");
        let results = store.filter(&filter);

        assert_eq!(results.len(), 34);
    }

    #[test]
    fn test_filter_in() {
        let mut store = create_test_store();
        store.build_index("category");

        let filter = FilterCondition::is_in(
            "category",
            vec![
                MetadataValue::String("electronics".into()),
                MetadataValue::String("books".into()),
            ],
        );
        let results = store.filter(&filter);

        // electronics: 34, books: 33
        assert_eq!(results.len(), 67);
    }

    #[test]
    fn test_filter_range() {
        let store = create_test_store();

        let filter = FilterCondition::range("price", 100.0f64, 200.0f64);
        let results = store.filter(&filter);

        // prices 100, 110, 120, ..., 200 = 11 items
        assert_eq!(results.len(), 11);
    }

    #[test]
    fn test_filter_and() {
        let store = create_test_store();

        let filter = FilterCondition::and(vec![
            FilterCondition::eq("category", "electronics"),
            FilterCondition::eq("in_stock", true),
        ]);
        let results = store.filter(&filter);

        // electronics (i % 3 == 0) AND in_stock (i % 2 == 0) = i % 6 == 0
        assert_eq!(results.len(), 17); // 0, 6, 12, ..., 96
    }

    #[test]
    fn test_matches() {
        let store = create_test_store();

        let filter = FilterCondition::eq("category", "electronics");
        assert!(store.matches(0, &filter));
        assert!(!store.matches(1, &filter));
    }

    #[test]
    fn test_bitmap_operations() {
        let mut store = create_test_store();
        store.build_index("category");

        let filter = FilterCondition::eq("category", "electronics");
        let bitmap = store.filter_bitmap(&filter);

        assert_eq!(bitmap.len(), 34);
    }
}
