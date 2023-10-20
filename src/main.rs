use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;
extern crate serde;
extern crate serde_derive;
use serde::{Serialize, Deserialize};

struct PhraseMatcher {
    tokenizer: fn(&str) -> Vec<String>,
    model_dir: String,
    vocab: HashMap<String, usize>,
    inv_vocab: HashMap<usize, String>,
    patterns: Patterns,
}

#[derive(Serialize, Deserialize)]
struct Patterns {
    lengths: HashSet<usize>,
    b_ints: HashSet<usize>,
    e_ints: HashSet<usize>,
    checksums: HashSet<(u32, u32)>,
}

impl Patterns {
    fn new() -> Self {
        Patterns {
            lengths: HashSet::new(),
            b_ints: HashSet::new(),
            e_ints: HashSet::new(),
            checksums: HashSet::new(),
        }
    }
}

impl PhraseMatcher {
    fn new(
        model_dir: &str,
        pattern_file: Option<&str>,
        vocab_file: Option<&str>,
        max_len: usize,
        tokenizer: fn(&str) -> Vec<String>,
    ) -> Self {
        let mut matcher = PhraseMatcher {
            tokenizer,
            model_dir: model_dir.to_string(),
            vocab: HashMap::<String, usize>::new(),
            inv_vocab: HashMap::<usize, String>::new(),
            patterns: Patterns::new(),
        };

        if !Path::new(model_dir).exists() {
            fs::create_dir_all(model_dir).expect("Failed to create model directory");
        }

        if let Some(pattern_file) = pattern_file {
            if let Some(vocab_file) = vocab_file {
                matcher.read_vocab(vocab_file);
            } else {
                matcher.build_vocab(pattern_file);
            }
            matcher.compile(pattern_file, max_len);
        } else {
            matcher.load_saved_data();
        }

        matcher
    }

    fn read_vocab(&mut self, fname: &str) {
        println!("Reading vocab file...");
        let mut wc = HashMap::<String, usize>::new();

        if let Ok(file) = fs::File::open(fname) {
            let reader = io::BufReader::new(file);
            for line in reader.lines() {
                if let Ok(line) = line {
                    let parts = (self.tokenizer)(&line.to_lowercase().trim());
                    if let Some(word) = parts.get(0) {
                        wc.insert(word.to_string(), wc.len());
                    }
                }
            }
        }

        let n_vocab = wc.len();
        self.vocab = wc.clone();

        let mut sorted_wc: Vec<_> = wc.into_iter().collect();
        sorted_wc.sort_by(|a, b| b.1.cmp(&a.1));

        for (idx, (word, _)) in sorted_wc.iter().enumerate() {
            self.inv_vocab.insert(idx, word.clone());
        }

        let vocab_file = format!("{}/vocab.p", self.model_dir);
        let vocab_file = fs::File::create(vocab_file).expect("Failed to create vocab file");
        if let Err(err) = bincode::serialize_into(vocab_file, &self.vocab) {
            eprintln!("Failed to serialize vocab: {:?}", err);
        }
        println!("Vocab size: {}", n_vocab);
    }

    fn build_vocab(&mut self, fname: &str) {
        println!("Start building vocab...");
        let mut wc = HashMap::<String, usize>::new();

        if let Ok(file) = fs::File::open(fname) {
            let reader = io::BufReader::new(file);
            for line in reader.lines() {
                if let Ok(line) = line {
                    for word in (self.tokenizer)(&line.to_lowercase().trim()) {
                        *wc.entry(word.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }

        self.vocab = wc.clone();

        let mut sorted_wc: Vec<_> = wc.into_iter().collect();
        sorted_wc.sort_by(|a, b| b.1.cmp(&a.1));

        for (idx, (word, _)) in sorted_wc.iter().enumerate() {
            self.inv_vocab.insert(idx, word.clone());
        }


        let vocab_file = format!("{}/vocab.p", self.model_dir);
        let vocab_file = fs::File::create(vocab_file).expect("Failed to create vocab file");
        if let Err(err) = bincode::serialize_into(vocab_file, &self.vocab) {
            eprintln!("Failed to serialize vocab: {:?}", err);
        }
        println!("Vocab size: {}", self.vocab.len());
    }


    fn compile(&mut self, fname: &str, max_len: usize) {
        println!("Start compiling patterns...");
        self.patterns = Patterns::new();

        if let Ok(file) = fs::File::open(fname) {
            let reader = io::BufReader::new(file);
            for (i, pat) in reader.lines().enumerate() {
                if i % 100000 == 0 {
                    println!("Processing input patterns: {}", i);
                }

                let pat = pat.expect("Failed to read pattern");
                let p_arr: Vec<_> = pat.trim().split_whitespace().collect();
                let p_len = p_arr.len();

                if p_len > max_len {
                    continue;
                }

                let mut p_ints = Vec::new();
                for t in &p_arr {
                    if let Some(&v) = self.vocab.get(*t) {
                        p_ints.push(v);
                    } else {
                        p_ints.clear();
                        break;
                    }
                }

                if p_ints.is_empty() {
                    continue;
                }

                let p_c = self.crc32(&p_arr.join(" "));
                let p_f = self.fletcher(&p_ints);

                self.patterns.lengths.insert(p_len);
                self.patterns.b_ints.insert(p_ints[0]);
                self.patterns.e_ints.insert(p_ints[p_len - 1]);
                self.patterns.checksums.insert((p_c, p_f));
            }
        }

        let patterns_file = format!("{}/patterns.p", self.model_dir);
        let patterns_file = fs::File::create(patterns_file).expect("Failed to create patterns file");
        if let Err(err) = bincode::serialize_into(patterns_file, &self.patterns) {
            eprintln!("Failed to serialize patterns: {:?}", err);
        }
    }

    fn crc32(&self, text: &str) -> u32 {
        let s = text.as_bytes();
        let crc = crc::crc32::checksum_ieee(s);
        crc % 0xFFFFFFFF
    }

    fn fletcher(&self, arr: &[usize]) -> u32 {
        let mut sum1: u32 = 0;
        let mut sum2: u32 = 0;
        for v in arr {
            sum1 = (sum1 + *v as u32) % 255;
            sum2 = (sum2 + sum1) % 255;
        }
        (sum1 * 256) + sum2
    }

    fn match_phrase(&self, sentence: &str, remove_subset: bool) -> Vec<String> {
        let tok = (self.tokenizer)(sentence.trim());
        let mut tok_ints = Vec::new();
        for t in &tok {
            if let Some(&v) = self.vocab.get(t.as_str()) {
                tok_ints.push(v);
            } else {
                tok_ints.push(0);
            }
        }
        let tok_len = tok_ints.len();
        let mut candidates = HashSet::new();

        for (i, &b_int) in tok_ints.iter().enumerate() {
            if !self.patterns.b_ints.contains(&b_int) {
                continue;
            }

            for &p_len in &self.patterns.lengths {
                let j = i + p_len - 1;
                if j + 1 > tok_len {
                    continue;
                }

                let p_ints = &tok_ints[i..=j];
                if p_ints.contains(&0) {
                    continue;
                }

                let e_int = tok_ints[j];
                if !self.patterns.e_ints.contains(&e_int) {
                    continue;
                }

                let p_c = self.crc32(&tok[i..=j].join(" "));
                let p_f = self.fletcher(&p_ints);

                if self.patterns.checksums.contains(&(p_c, p_f)) {
                    candidates.insert((i, j));
                }
            }
        }

        let mut results = Vec::new();

        if remove_subset {
            let mut to_remove = Vec::new();

            for &(i, j) in &candidates {
                let mut should_remove = false;

                for &(ii, jj) in &candidates {
                    if i == ii && j == jj {
                        continue;
                    }

                    if ii <= i && j <= jj {
                        should_remove = true;
                        break;
                    }
                }

                if should_remove {
                    to_remove.push((i, j));
                }
            }

            for &(i, j) in &to_remove {
                candidates.remove(&(i, j));
            }
        }

        for &(i, j) in &candidates {
            results.push(tok[i..=j].join(" "));
        }

        results
    }

    fn load_saved_data(&mut self) {
        let vocab_file = format!("{}/vocab.p", self.model_dir);
        let patterns_file = format!("{}/patterns.p", self.model_dir);

        if let Ok(vocab_file) = fs::File::open(vocab_file) {
            if let Ok(vocab) = bincode::deserialize_from(vocab_file) {
                self.vocab = vocab;
            }
        }

        if let Ok(patterns_file) = fs::File::open(patterns_file) {
            if let Ok(patterns) = bincode::deserialize_from(patterns_file) {
                self.patterns = patterns;
            }
        }
    }
}

fn main() {
    let model_dir = "model_dir";
    let pattern_file = Some("patterns.txt");
    let vocab_file = None;
    let max_len = 10;
    let tokenizer = |x: &str| x.split_whitespace().map(|s| s.to_string()).collect::<Vec<_>>();

    let phrase_matcher = PhraseMatcher::new(model_dir, pattern_file, vocab_file, max_len, tokenizer);

    let sentence = "menurut analisa squawka , mu adalah satu di antara lima kesebelasan dengan kesalahan defensif terbesar di epl musim lalu -- walau hanya tiga gol yang masuk ke gawang mereka dari sejumlah kesalahan itu .";
    let remove_subset = false;
    let matches = phrase_matcher.match_phrase(sentence, remove_subset);
    println!("Matches: {:?}", matches);
}
