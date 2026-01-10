use crate::step3_mcmc::{McmcContext, ChainState, run_mini_em};
use anyhow::{Context, Result};
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::{Gamma, Distribution};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Parses NCBI names.dmp file: ID | Name | UniqueName | Class |
/// Returns a map of TaxonID (String) -> Scientific Name
pub fn load_taxonomy_names(path: &str) -> Result<HashMap<String, String>> {
    println!("Loading taxonomy names from: {}", path);
    let file = File::open(path).context("Failed to open names.dmp")?;
    let reader = BufReader::new(file);
    let mut map = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
        if parts.len() >= 4 {
            let id = parts[0];
            let name = parts[1];
            let class = parts[3];

            if class == "scientific name" {
                map.insert(id.to_string(), name.to_string());
            }
        }
    }
    println!("Loaded {} scientific names.", map.len());
    Ok(map)
}

// Holds Summary Statistics for a Species
pub struct AbundanceStats {
    pub mean: f64,
    pub ci_low: f64,  // 2.5th percentile
    pub ci_high: f64, // 97.5th percentile
}

pub fn run_inference(
    ctx: &McmcContext,
    chains: &[ChainState],
    burnin_ratio: f64,
    output_prefix: &str,
    taxonomy_map: Option<&HashMap<String, String>>,
    index_to_read_name: &[String], 
    output_posterior: bool,
    gibbs_iter: usize,
    gibbs_burnin: usize,
) -> Result<()> {
    println!("\n=== STEP 4: INFERENCE & REPORTING ===");

    // 1. Dynamic Unknown Integration (Median of all chains)
    let mut floors: Vec<f64> = chains.iter().map(|c| c.current_unk_prob).collect();
    let mid = floors.len() / 2;
    floors.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    let learned_floor = floors[mid];
    println!("Learned Unknown Probability Floor (Median): {:.4e}", learned_floor);

    // 2. Select Best Chain (Coldest)
    let cold_chain = &chains[0];
    println!("Analyzing Cold Chain (ID: {}) with {} species.", cold_chain.id, cold_chain.species_set.len());

    // 3. Export MCMC Trace (QC)
    let trace_path = format!("{}_mcmc_trace.tsv", output_prefix);
    export_trace(&trace_path, cold_chain, burnin_ratio)?;

    // 4. Bayes Factors (Parallel)
    println!("Computing Bayes Factors for {} species...", cold_chain.species_set.len());
    
    let mut candidate_indices: Vec<usize> = cold_chain.species_set.iter().cloned().collect();
    candidate_indices.sort_unstable(); // Deterministic

    let h1_log_l = cold_chain.current_log_likelihood;

    let bayes_factors: HashMap<usize, f64> = candidate_indices.par_iter().map(|&sp_idx| {
        let mut h0_set = cold_chain.species_set.clone();
        h0_set.remove(&sp_idx);
        
        let n_h0 = h0_set.len();
        let mut init_abund = HashMap::new();
        if n_h0 > 0 {
            let mut rng = rand::rng();
            let gamma = Gamma::new(1.0, 1.0).unwrap();
            let mut sum = 0.0;
            let mut samples = Vec::with_capacity(n_h0);
            for _ in 0..n_h0 {
                let x = gamma.sample(&mut rng);
                samples.push(x);
                sum += x;
            }
            for (i, &id) in h0_set.iter().enumerate() {
                init_abund.insert(id, samples[i] / sum);
            }
        }

        let (h0_log_l, _, _) = run_mini_em(
            ctx, 
            &h0_set, 
            &init_abund, 
            learned_floor, 
            10 
        );

        let bf = h1_log_l - h0_log_l + ctx.lpenalty; 
        let log10_bf = bf / 10.0_f64.ln(); 
        
        (sp_idx, log10_bf)
    }).collect();

    // 5. Final Gibbs Sampler (Read Assignments + CI)
    println!("Running Final Gibbs Sampler ({} iter + {} burnin)...", gibbs_iter, gibbs_burnin);
    let (final_stats, read_assignments, count_history) = run_gibbs_sampler(
        ctx,
        &cold_chain.species_set,
        learned_floor,
        gibbs_iter,
        gibbs_burnin,
    );

    // 6. Export Main Results
    let results_path = format!("{}_results.tsv", output_prefix);
    export_summary(
        &results_path, 
        ctx, 
        &final_stats, 
        &bayes_factors, 
        learned_floor,
        taxonomy_map
    )?;

    // 7. Export Read Assignments (Always output if names exist)
    if !index_to_read_name.is_empty() {
        let reads_path = format!("{}_read_assignments.tsv", output_prefix);
        export_read_assignments(
            &reads_path,
            ctx,
            &read_assignments,
            index_to_read_name,
            taxonomy_map,
            &cold_chain.species_set
        )?;
    }

    // 8. Export Posterior Samples (Conditional)
    if output_posterior {
        let post_path = format!("{}_posterior_samples.tsv", output_prefix);
        export_posterior_samples(
            &post_path,
            ctx,
            &count_history,
            &cold_chain.species_set
        )?;
    }

    Ok(())
}

fn run_gibbs_sampler(
    ctx: &McmcContext,
    species_set: &HashSet<usize>,
    unk_prob: f64,
    iterations: usize,
    burnin: usize,
) -> (HashMap<usize, AbundanceStats>, Vec<(usize, f64)>, Vec<Vec<f64>>) {
    
    let mut active_indices: Vec<usize> = species_set.iter().cloned().collect();
    active_indices.sort_unstable();

    let num_active = active_indices.len();
    let mut abundances = vec![1.0 / (num_active + 1) as f64; num_active + 1];

    let mut abund_history: Vec<Vec<f64>> = Vec::with_capacity(iterations);
    let mut count_history: Vec<Vec<f64>> = Vec::with_capacity(iterations);

    for i in 0..(iterations + burnin) {
        let mut counts = vec![0.0; num_active + 1];

        let iter_counts: Vec<Vec<f64>> = ctx.matrix.row_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|(row_idx, row_vec)| {
                let mut rng = rand::rng();
                let mut local_counts = vec![0.0; num_active + 1];
                let mut weights = Vec::with_capacity(num_active + 1);
                let mut total_w = 0.0;

                for j in 0..num_active {
                    let col_idx = active_indices[j];
                    let mut p_val = 0.0;
                    for (c, &v) in row_vec.col_indices().iter().zip(row_vec.values()) {
                        if *c == col_idx { p_val = v; break; }
                    }
                    let w = p_val * abundances[j];
                    weights.push(w);
                    total_w += w;
                }

                let w_unk = unk_prob * abundances[num_active];
                weights.push(w_unk);
                total_w += w_unk;

                let sample = rng.random::<f64>() * total_w;
                let mut cumulative = 0.0;
                let mut chosen_idx = num_active;

                for (k, &w) in weights.iter().enumerate() {
                    cumulative += w;
                    if sample <= cumulative { chosen_idx = k; break; }
                }

                local_counts[chosen_idx] += ctx.read_weights[*row_idx];
                local_counts
            })
            .collect();

        for c in iter_counts {
            for (idx, val) in c.iter().enumerate() { counts[idx] += val; }
        }

        let alpha: Vec<f64> = counts.iter().map(|&n| n + 1.0).collect();
        let mut rng = rand::rng(); 
        let mut samples = Vec::with_capacity(num_active + 1);
        let mut sum = 0.0;
        for &a in &alpha {
            let x = Gamma::new(a, 1.0).unwrap().sample(&mut rng);
            samples.push(x);
            sum += x;
        }
        for x in &mut samples { *x /= sum; }
        abundances = samples;

        if i >= burnin {
            abund_history.push(abundances.clone());
            count_history.push(counts);
        }
    }

    let mut final_stats_map = HashMap::new();
    let mut mean_abundances_vec = vec![0.0; num_active + 1];

    for i in 0..num_active {
        let mut vals: Vec<f64> = abund_history.iter().map(|vec| vec[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean: f64 = vals.iter().sum::<f64>() / iterations as f64;
        let ci_low = vals[(iterations as f64 * 0.025).round() as usize].clone();
        let ci_high = vals[(iterations as f64 * 0.975).round() as usize].clone();

        final_stats_map.insert(active_indices[i], AbundanceStats { mean, ci_low, ci_high });
        mean_abundances_vec[i] = mean;
    }
    
    mean_abundances_vec[num_active] = abund_history.iter().map(|v| v[num_active]).sum::<f64>() / iterations as f64;

    let read_assignments: Vec<(usize, f64)> = ctx.matrix.row_iter()
        .map(|row_vec| {
            let mut best_idx = num_active;
            let mut best_prob = unk_prob * mean_abundances_vec[num_active];
            let mut total_prob = best_prob;

            for (i, &col_idx) in active_indices.iter().enumerate() {
                let mut p_val = 0.0;
                for (c, &v) in row_vec.col_indices().iter().zip(row_vec.values()) {
                    if *c == col_idx { p_val = v; break; }
                }
                let term = p_val * mean_abundances_vec[i];
                total_prob += term;
                if term > best_prob {
                    best_prob = term;
                    best_idx = i;
                }
            }
            (best_idx, if total_prob > 0.0 { best_prob / total_prob } else { 1.0 })
        })
        .collect();

    (final_stats_map, read_assignments, count_history)
}

fn export_posterior_samples(
    path: &str,
    ctx: &McmcContext,
    history: &[Vec<f64>],
    active_indices_set: &HashSet<usize>,
) -> Result<()> {
    println!("Exporting posterior distribution to: {}", path);
    let mut file = BufWriter::new(File::create(path)?);
    let mut active_vec: Vec<usize> = active_indices_set.iter().cloned().collect();
    active_vec.sort_unstable();

    write!(file, "Iteration")?;
    for &idx in &active_vec { write!(file, "\t{}", ctx.taxons[idx])?; }
    writeln!(file, "\tUnknown")?;

    for (i, counts) in history.iter().enumerate() {
        write!(file, "{}", i)?;
        for val in counts { write!(file, "\t{:.2}", val)?; }
        writeln!(file)?;
    }
    Ok(())
}

fn export_summary(
    path: &str,
    ctx: &McmcContext,
    abundances: &HashMap<usize, AbundanceStats>,
    bayes_factors: &HashMap<usize, f64>,
    floor: f64,
    taxonomy_map: Option<&HashMap<String, String>>,
) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "TaxonID\tScientificName\tMeanAbundance\tCI_Lower\tCI_Upper\tEstimatedReads\tLog10BF\tPosterior")?;
    let total_reads: f64 = ctx.read_weights.iter().sum();
    for (&idx, stats) in abundances {
        let tid = &ctx.taxons[idx];
        let name = taxonomy_map.and_then(|m| m.get(tid)).map(|s| s.as_str()).unwrap_or("Unknown");
        writeln!(file, "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.2}\t{:.2}\t1.00", 
            tid, name, stats.mean, stats.ci_low, stats.ci_high, stats.mean * total_reads, bayes_factors.get(&idx).unwrap_or(&0.0))?;
    }
    writeln!(file, "# Unknown_Bin_Probability_Floor: {:.4e}", floor)?;
    Ok(())
}

fn export_read_assignments(
    path: &str,
    ctx: &McmcContext,
    assignments: &[(usize, f64)],
    read_names: &[String],
    taxonomy_map: Option<&HashMap<String, String>>,
    active_indices_set: &HashSet<usize>,
) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "ReadName\tAssignedTaxonID\tAssignedName\tProbability")?;
    let mut active_vec: Vec<usize> = active_indices_set.iter().cloned().collect();
    active_vec.sort_unstable();
    for (r_idx, (best_idx, prob)) in assignments.iter().enumerate() {
        if *best_idx >= active_vec.len() {
            writeln!(file, "{}\tUnknown\tUnknown\t{:.4}", read_names[r_idx], prob)?;
        } else {
            let tid = &ctx.taxons[active_vec[*best_idx]];
            let name = taxonomy_map.and_then(|m| m.get(tid)).map(|s| s.as_str()).unwrap_or("Unknown");
            writeln!(file, "{}\t{}\t{}\t{:.4}", read_names[r_idx], tid, name, prob)?;
        }
    }
    Ok(())
}

fn export_trace(path: &str, chain: &ChainState, ratio: f64) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "Iteration\tLogLikelihood\tMoveType")?;
    let start = (chain.history.len() as f64 * ratio) as usize;
    for rec in &chain.history[start..] {
        writeln!(file, "{}\t{:.4}\t{}", rec.iter, rec.log_likelihood, rec.move_type)?;
    }
    Ok(())
}