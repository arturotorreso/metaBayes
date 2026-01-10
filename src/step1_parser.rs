use anyhow::{Context, Result};
use noodles::bam;
use noodles::sam::alignment::record::data::field::Tag;
use noodles::sam::alignment::record::cigar::op::Kind; 
use statrs::function::gamma::{gamma_lr, ln_gamma}; 
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::collections::HashMap;

use crate::step2_reduce::{MatrixEntry, TaxonomyParser};

fn calculate_total_lambda(scores: &[u8]) -> f64 {
    let mut lambda = 0.0;
    for &score in scores {
        let q = score as f64;
        let p_err = 10f64.powf(-q / 10.0);
        lambda += p_err;
    }
    lambda
}

fn calculate_r_score(k: u32, lambda: f64, genome_len: u64) -> f64 {
    if k == 0 {
        return -(genome_len as f64).ln();
    }
    
    let safe_lambda = if lambda < 1e-100 { 1e-100 } else { lambda };
    let k_val = k as f64;
    
    let prob_tail = gamma_lr(k_val, safe_lambda);
    
    let log_pij = if prob_tail > 0.0 {
        (prob_tail / (genome_len as f64)).ln()
    } else {
        // Log-Space Approximation
        let log_poisson = -safe_lambda + (k_val * safe_lambda.ln()) - ln_gamma(k_val + 1.0);
        log_poisson - (genome_len as f64).ln()
    };

    if log_pij < -700.0 { -700.0 } else { log_pij }
}

pub fn process_bam<P: AsRef<Path>>(path: P) -> Result<(Vec<MatrixEntry>, usize, TaxonomyParser, Vec<String>, f64)> {
    
    let file = File::open(path).context("Failed to open BAM file")?;
    let mut reader = bam::io::Reader::new(BufReader::new(file));
    let header = reader.read_header().context("Failed to read BAM header")?;
    let references = header.reference_sequences();

    println!("--- Parsing BAM (Assuming Name-Sorted) ---");

    // --- 1. Calculate Median Genome Length ---
    let mut ref_lengths: Vec<u64> = references
        .values()
        .map(|rs| rs.length().get() as u64) // Explicit get() and cast
        .collect();
    
    let median_len = if ref_lengths.is_empty() {
        eprintln!("Warning: No reference sequences found in BAM header. Using default 284332.0");
        284332.0 
    } else {
        ref_lengths.sort_unstable();
        let mid = ref_lengths.len() / 2;
        if ref_lengths.len() % 2 == 0 {
            (ref_lengths[mid - 1] + ref_lengths[mid]) as f64 / 2.0
        } else {
            ref_lengths[mid] as f64
        }
    };
    println!("Detected Median Genome Length: {:.0} bp", median_len);

    let mut entries: Vec<MatrixEntry> = Vec::new();
    let mut parser = TaxonomyParser::new()?;
    let mut index_to_read_name: Vec<String> = Vec::new();
    let mut next_read_idx = 0;

    // Pre-build Reference Name Map
    let ref_names: Vec<String> = references
        .keys()
        .map(|k| k.to_string())
        .collect();

    // State
    let mut last_name: Vec<u8> = Vec::new();
    let mut cached_lambda: f64 = 0.0;
    let mut cached_len: u64 = 0; 
    let unknown_name: &[u8] = b"unknown";

    let mut read_buffer: HashMap<usize, f64> = HashMap::new();
    let mut current_read_row_idx = 0; 

    // Helper to flush buffer
    let flush_buffer = |entries: &mut Vec<MatrixEntry>, buffer: &mut HashMap<usize, f64>, r_idx: usize| {
        for (t_idx, log_p) in buffer.drain() {
            entries.push(MatrixEntry { read_idx: r_idx, taxon_idx: t_idx, log_prob: log_p });
        }
    };

    for result in reader.record_bufs(&header) {
        let record = result.context("Failed to parse a record")?;
        let current_name_bytes = record.name().map(|n| n.as_ref()).unwrap_or(unknown_name);
        
        // --- 2. Read ID Management (Streaming Mode) ---
        // Since we assume Name-Sorted, any change in name means a new read ID.
        let is_name_change = current_name_bytes != last_name.as_slice();
        
        if is_name_change {
            // Flush previous read if it existed
            if !last_name.is_empty() {
                flush_buffer(&mut entries, &mut read_buffer, current_read_row_idx);
            }

            // Assign new ID
            current_read_row_idx = next_read_idx;
            next_read_idx += 1;
            
            // Store name for output (Step 4 needs names)
            index_to_read_name.push(String::from_utf8_lossy(current_name_bytes).to_string());

            // Reset Cache
            cached_lambda = -1.0; 
            cached_len = 0;
            last_name = current_name_bytes.to_vec();
        }

        // --- 3. Resolve Taxon ---
        let (taxon_idx, genome_len) = if let Some(ref_id) = record.reference_sequence_id() {
             if let Some(name_str) = ref_names.get(ref_id) {
                 match parser.get_taxon_index(name_str) {
                     Some(idx) => {
                         if let Some(map_record) = references.get(name_str.as_bytes()) {
                             (idx, map_record.length().get() as u64)
                         } else { continue; }
                     },
                     None => continue, 
                 }
             } else { continue; }
        } else { continue; };

        // --- 4. Calculate Lambda (Phred/Length) ---
        let raw_len = if !record.sequence().is_empty() {
             record.sequence().len()
        } else {
             let mut l = 0;
             for op in record.cigar().as_ref().iter() {
                 match op.kind() {
                     Kind::Match | Kind::Insertion | Kind::SoftClip | Kind::SequenceMatch | Kind::SequenceMismatch => {
                         l += op.len();
                     },
                     _ => {}
                 }
             }
             l
        } as u64;

        let raw_quality = record.quality_scores();
        let current_lambda = if !raw_quality.is_empty() {
            let val = calculate_total_lambda(raw_quality.as_ref());
            cached_lambda = val;
            cached_len = raw_len;
            val
        } 
        else if cached_lambda >= 0.0 && raw_len == cached_len {
            cached_lambda
        }
        else {
            0.03 * (raw_len as f64)
        };

        let lambda_final = current_lambda.max(0.01 * (raw_len as f64));

        // --- 5. Get Mismatches (k) ---
        let data = record.data();
        let nm_value = data.get(&Tag::EDIT_DISTANCE).or_else(|| data.get(&Tag::ALIGNMENT_HIT_COUNT)); 
        let k = match nm_value {
            Some(value) => value.as_int().unwrap_or(0) as u32,
            None => 0, 
        };

        // --- 6. Score Alignment ---
        let new_log_pij = calculate_r_score(k, lambda_final, genome_len);

        // Keep best score per read-taxon pair
        read_buffer.entry(taxon_idx)
            .and_modify(|e| *e = e.max(new_log_pij))
            .or_insert(new_log_pij);
    }
    
    // Flush final read
    flush_buffer(&mut entries, &mut read_buffer, current_read_row_idx);

    println!("Parsed {} entries for {} unique reads.", entries.len(), next_read_idx);
    Ok((entries, next_read_idx, parser, index_to_read_name, median_len))
}