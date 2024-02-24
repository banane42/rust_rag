use std::{collections::HashMap, fmt::Display, path::PathBuf};

use crate::bow;

pub struct Corpus {
    documents: Vec<Document>,
    terms: HashMap<String, f32>,
}

impl Corpus {

    pub fn new() -> Self {
        Self { 
            documents: Vec::new(), 
            terms: HashMap::new()
        }
    }

    pub fn push_document(&mut self, doc: Document) {
        doc.bow.keys().for_each(|doc_term| {
            if !self.terms.contains_key(doc_term) {
                self.terms.insert(doc_term.clone(), 0.0);
            }
        });
        
        self.documents.push(doc)
    }

    pub fn build(&mut self) {
        println!("Building Corpus");
        println!("Calculating term idf...");
        self.terms.shrink_to_fit();
        // Calculate idf for each term in the Corpus
        for (term, val) in self.terms.iter_mut() {
            let mut docs_containing_count: usize = 0;
            for doc in self.documents.iter() {
                if doc.contains_term(term) {
                    docs_containing_count += 1;
                }
            }

            // Calculate idf
            // log10(N / D)
            // Where N is the number of documents
            // and D is the number of documents containing the specific term
            let idf = ((1.0 + self.documents.len() as f32) / (1.0 + docs_containing_count as f32)).log10();
            *val = idf;
        }

        // Assign vector to document
        for doc in self.documents.iter_mut() {
            let mut vec = vec![0.0; self.terms.len()];
            for (i, (term, idf)) in self.terms.iter().enumerate() {
                let tf = *doc.bow.get(term).unwrap_or(&0) as f32;
                vec[i] = tf * *idf;
            }
            doc.tf_idf_score_vector = vec;
        }
    }

    pub fn query(&self, query: String) -> usize {
        let q_bow = bow::string_to_bow(query);
        
        // Calculate tf_idf vector for query
        let mut q_vec: Vec<f32> = vec![0.0; self.terms.len()];
        for (i, (term, idf)) in self.terms.iter().enumerate() {
            let tf = *q_bow.get(term).unwrap_or(&0) as f32;
            q_vec[i] = tf * *idf;
        }

        // Do Cos(Theta) of q_vec vs all document vectors
        // Cos(Theta) = (doc_vec dot q_vec) / (doc_vec.mag * q_vec.mag)
        let mut best_cos = f32::MIN;
        let mut best_doc_i: usize = 0;
        for (i, doc) in self.documents.iter().enumerate() {
            let cos = cos_theta(&q_vec, &doc.tf_idf_score_vector);
            if cos > best_cos {
                best_cos = cos;
                best_doc_i = i;
            }
        }

        return best_doc_i;
    }

    pub fn get_doc_path(&self, index: usize) -> Option<PathBuf> {
        return match self.documents.get(index) {
            Some(doc) => Some(doc.path.clone()),
            None => None,
        }
    }

}

#[derive(Debug)]
pub struct Document {
    pub path: PathBuf,
    pub bow: HashMap::<String, u32>,
    tf_idf_score_vector: Vec<f32>
}

impl Document {
    pub fn new(path: PathBuf, bow: HashMap<String, u32>) -> Self {
        Self { 
            path,
            bow, 
            tf_idf_score_vector: Vec::new() 
        }
    }

    pub fn contains_term(&self, term: &String) -> bool {
        self.bow.contains_key(term)   
    }
}

impl Display for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path.to_str().unwrap_or(""))
    }
}

fn cos_theta(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let dot = dot_product(a, b).expect("Vecs not same size");
    let mag = mag(a) * mag(b);

    return dot / mag;
}

fn dot_product(a: &Vec<f32>, b: &Vec<f32>) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    let mut val = 0.0;
    for i in 0..a.len() {
        val += a[i] * b[i]
    }

    return Some(val);
}

fn mag(a: &Vec<f32>) -> f32 {
    let mut sum_sqr = 0.0;

    a.iter().for_each(|x| {
        sum_sqr += x * x;
    });

    return sum_sqr.sqrt();
}