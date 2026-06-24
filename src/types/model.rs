use serde::{Deserialize, Serialize};

/// Response that list available models.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelsListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}