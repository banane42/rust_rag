use std::{error::Error, fs::{self, read_to_string}, path::Path};

use document::Document;
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest};

use crate::document::Corpus;

mod bow;
mod document;

// fn main() {

//     let mut corpus = Corpus::new();

//     read_docs(&mut corpus).expect("Error reading corpus docs");

//     corpus.build();

//     // let doc_i = corpus.query(read_to_string(Path::new("./corpus/core_concepts_missions.txt")).unwrap());
//     let doc_i = corpus.query("What is the game of Warhamer 40,000?".to_string());
//     println!("Query Doc: {:?}", corpus.get_doc_path(doc_i));

//     // let intro_path = Path::new("./corpus/introduction.txt");
//     // let intro_bow = bow::document_to_bow(intro_path).unwrap();
//     // println!("{:#?}", intro_bow);
// }

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let mut corpus = Corpus::new();
    read_docs(&mut corpus)?;
    corpus.build();

    // By default it will connect to localhost:11434
    let ollama = Ollama::default();

    query_generate("What are some core concepts that I will have to know to play?".to_string(), &ollama, &corpus).await?;

    // test_generate(&ollama).await;
    Ok(())
}

async fn query_generate(query: String, ollama: &Ollama, corpus: &Corpus) -> Result<(), Box<dyn Error>>{
    
    let mut prompt = String::from("You are a bot that answers questions to the game Warhammer 40,000. Keep your answers short and succinct.");
    let doc = corpus.get_doc_path(corpus.query(query.clone())).expect("Query did not return a valid index");

    let doc_contents = read_to_string(doc.as_path())?;
    prompt.push_str("\nThe relevant rules section is\n");
    prompt.push_str(&doc_contents);

    prompt.push_str("\nThe user's question is\n");
    prompt.push_str(&query);
    prompt.push_str("\nAnswer the user's question based on the relevant rules section provided earlier.");

    println!("Query: {}\nDocument: {:?}", query, doc);

    let model = "tinyllama:latest".to_string();

    let result = ollama.generate(GenerationRequest::new(
        model, prompt
    )).await;

    println!("{}", result.unwrap().response);

    Ok(())
}

async fn test_generate(ollama: &Ollama) {
    // let model = "llama2:latest".to_string();
    let model = "tinyllama:latest".to_string();
    let prompt = "What did I ask you last?".to_string();

    let res = ollama.generate(GenerationRequest::new(model, prompt)).await;
    
    println!("{}", res.unwrap().response);

    // if let Ok(res) = res {
    //     println!("{}", res.response);
    // }
}

fn read_docs(corpus: &mut Corpus) -> Result<(), Box<dyn Error>> {
    let corpus_path = Path::new("./corpus");
    let entries = fs::read_dir(corpus_path)?;

    for entry in entries {
        let entry = entry?;
        if entry.path().is_file() && entry.path().extension().map_or(false, |ext| ext == "txt") {
            println!("Creating Document {:?}", entry.path());
            let bow = bow::document_to_bow(entry.path().as_path())?;
            let doc = Document::new(entry.path(), bow);
            corpus.push_document(doc);
        }
    }

    Ok(())
}