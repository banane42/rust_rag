use std::{collections::HashMap, fs::read_to_string, io::Error, path::Path};

pub fn document_to_bow(path: &Path) -> Result<HashMap<String, u32>, Error> {
    let doc_string = read_to_string(path)?;

    let bow_map = string_to_bow(doc_string);

    Ok(bow_map)
}

pub fn string_to_bow(string: String) -> HashMap<String, u32> {
    let mut bow_map = HashMap::<String, u32>::new();
    string.split_ascii_whitespace().for_each(|word| {
        let term = word
            .trim()
            .chars().filter(|c| !c.is_ascii_punctuation()).collect::<String>()
            .to_lowercase();

        match bow_map.get_mut(&term) {
            Some(count) => {
                *count += 1;
            },
            None => {
                bow_map.insert(term, 1);
            }
        }
    });
    
    bow_map
}