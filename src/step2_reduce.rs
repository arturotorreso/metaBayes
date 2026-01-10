use anyhow::Result;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct MatrixEntry {
    pub read_idx: usize,
    pub taxon_idx: usize,
    pub log_prob: f64,
}

pub struct TaxonomyParser {
    pub name_to_id: HashMap<String, usize>,
    pub index_to_id: Vec<String>,
}

impl TaxonomyParser {
    pub fn new() -> Result<Self> {
        Ok(Self {
            name_to_id: HashMap::new(),
            index_to_id: Vec::new(),
        })
    }

    pub fn get_taxon_index(&mut self, name: &str) -> Option<usize> {
        let id_part = if name.starts_with("ti|") {
            name.split('|').nth(1)?
        } else {
            name 
        };

        if let Some(&idx) = self.name_to_id.get(id_part) {
            Some(idx)
        } else {
            let idx = self.index_to_id.len();
            self.name_to_id.insert(id_part.to_string(), idx);
            self.index_to_id.push(id_part.to_string());
            Some(idx)
        }
    }
}

pub struct Step2Result {
    pub matrix: CsrMatrix<f64>,
    pub reduced_taxons: Vec<String>, 
    pub reduced_abundances: Vec<f64>,
}

pub fn run_em_reduction(
    entries: Vec<MatrixEntry>,
    num_reads: usize,
    parser: TaxonomyParser,
    read_cutoff: usize,
    iterations: usize,
    verbose: bool,
) -> Result<Step2Result> {
    println!("Step 2: Filtering and building matrix...");

    // 1. Identification: Identify all species present in the input
    //    We do NOT filter by read_cutoff here anymore. We match R's logic:
    //    Run EM on *everything* that has at least 1 read.
    let mut species_present = HashSet::new();
    for e in &entries {
        species_present.insert(e.taxon_idx);
    }

    println!("Pre-EM: Found {} unique species in BAM input.", species_present.len());

    // 2. Remap Indices (Old ID -> New Matrix Column)
    let mut old_to_new_map = HashMap::new();
    let mut new_to_old_map = Vec::new();
    
    // Deterministic sort for stability
    let mut sorted_species: Vec<usize> = species_present.into_iter().collect();
    sorted_species.sort();

    for &old_idx in &sorted_species {
        let new_idx = new_to_old_map.len();
        old_to_new_map.insert(old_idx, new_idx);
        new_to_old_map.push(old_idx);
    }

    let num_species_pre = sorted_species.len();
    let mut coo = CooMatrix::new(num_reads, num_species_pre);

    for e in entries {
        if let Some(&new_col) = old_to_new_map.get(&e.taxon_idx) {
            coo.push(e.read_idx, new_col, e.log_prob);
        }
    }

    let csr = CsrMatrix::from(&coo);

    // 3. Run EM (On EVERYTHING)
    println!("Running EM for {} iterations...", iterations);
    let (em_abundances, final_iter) = run_em(&csr, iterations, verbose);

    println!("EM Converged/Stopped at iteration {}.", final_iter);

    // 4. Post-Filter: Match R Logic (Effective Count)
    //    R: ordered.species <- ordered.species[which(ordered.species$countReads >= read.cutoff), ]
    //    where countReads = round(abundance * total_reads)
    println!("Applying Post-EM filter (Effective Count >= {})...", read_cutoff);

    let total_reads_f64 = num_reads as f64;
    let mut survivor_indices = Vec::new();
    let mut reduced_taxons = Vec::new();
    let mut reduced_abundances = Vec::new();

    for (col_idx, &abund) in em_abundances.iter().enumerate() {
        // Match R's rounding logic
        let effective_count = (abund * total_reads_f64).round();
        
        if effective_count >= (read_cutoff as f64) {
            survivor_indices.push(col_idx);
            
            // Map back to original Taxon ID
            let old_idx = new_to_old_map[col_idx];
            let taxon_id = parser.index_to_id[old_idx].clone();
            
            reduced_taxons.push(taxon_id);
            reduced_abundances.push(abund);
        }
    }

    println!("Post-EM: Retained {} / {} species", reduced_taxons.len(), num_species_pre);

    // 5. Subset the Matrix for Step 3
    let final_matrix = subset_matrix_columns(&csr, &survivor_indices);

    Ok(Step2Result {
        matrix: final_matrix,
        reduced_taxons,
        reduced_abundances,
    })
}

// Helper: Efficiently create a new matrix containing only specific columns
fn subset_matrix_columns(input: &CsrMatrix<f64>, keep_cols: &[usize]) -> CsrMatrix<f64> {
    let num_rows = input.nrows();
    let num_new_cols = keep_cols.len();
    
    // Map: Old Column Index -> New Column Index
    let mut col_map = HashMap::new();
    for (new_idx, &old_idx) in keep_cols.iter().enumerate() {
        col_map.insert(old_idx, new_idx);
    }

    let mut coo = CooMatrix::new(num_rows, num_new_cols);

    for (row_idx, row_vec) in input.row_iter().enumerate() {
        for (col_idx, &val) in row_vec.col_indices().iter().zip(row_vec.values()) {
            if let Some(&new_col) = col_map.get(col_idx) {
                coo.push(row_idx, new_col, val);
            }
        }
    }

    CsrMatrix::from(&coo)
}

fn run_em(matrix: &CsrMatrix<f64>, iterations: usize, verbose: bool) -> (Vec<f64>, usize) {
    let _num_reads = matrix.nrows();
    let num_species = matrix.ncols();
    
    let mut abundances = vec![1.0 / num_species as f64; num_species];
    let mut next_abundances = vec![0.0; num_species];

    for iter in 0..iterations {
        next_abundances.fill(0.0);
        let mut total_log_l = 0.0;
        let mut diff = 0.0;

        for row in matrix.row_iter() {
            let mut max_val = -f64::INFINITY;
            for (col_idx, &log_p) in row.col_indices().iter().zip(row.values()) {
                let term = log_p + abundances[*col_idx].ln();
                if term > max_val { max_val = term; }
            }
            
            let mut sum_exp = 0.0;
            for (col_idx, &log_p) in row.col_indices().iter().zip(row.values()) {
                let term = log_p + abundances[*col_idx].ln();
                sum_exp += (term - max_val).exp();
            }
            
            let log_l_i = max_val + sum_exp.ln();
            total_log_l += log_l_i;

            for (col_idx, &log_p) in row.col_indices().iter().zip(row.values()) {
                let log_numerator = log_p + abundances[*col_idx].ln();
                let z_ij = (log_numerator - log_l_i).exp();
                next_abundances[*col_idx] += z_ij;
            }
        }

        let total_weight: f64 = next_abundances.iter().sum();
        for (j, val) in next_abundances.iter_mut().enumerate() {
            *val /= total_weight;
            diff += (*val - abundances[j]).abs();
        }

        abundances.copy_from_slice(&next_abundances);

        if verbose && (iter % 10 == 0 || iter == iterations - 1) {
            println!("EM Iter: {} | LogL: {:.2} | Diff: {:.6e}", iter, total_log_l, diff);
        }

        if diff < 1e-6 {
            return (abundances, iter + 1);
        }
    }

    (abundances, iterations)
}