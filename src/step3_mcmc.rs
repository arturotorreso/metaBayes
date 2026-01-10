use anyhow::Result;
use nalgebra_sparse::csr::CsrMatrix;
use rand::prelude::*;
use rand::distr::weighted::WeightedIndex;
use rand_distr::{Gamma, Distribution};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Barrier, RwLock};
use std::thread;

// ================================================================================================
// PART 1: DATA STRUCTURES & MINI-EM
// ================================================================================================

#[allow(dead_code)]
pub struct McmcContext {
    pub matrix: CsrMatrix<f64>,
    pub read_weights: Vec<f64>,
    pub taxons: Vec<String>,
    pub taxon_weights: Vec<f64>,
    pub median_genome_len: f64,
    pub read_support: usize,
    pub lpenalty: f64,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ChainState {
    pub id: usize,
    pub temperature: f64,
    pub species_set: HashSet<usize>,
    pub abundances: HashMap<usize, f64>, 
    pub current_unk_prob: f64, 
    pub current_log_likelihood: f64,
    pub moves_attempted: usize,
    pub moves_accepted: usize,
    pub swaps_attempted: usize,
    pub swaps_accepted: usize,
    pub history: Vec<ChainRecord>,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ChainRecord {
    pub iter: usize,
    pub log_likelihood: f64,
    pub move_type: String,
}

impl McmcContext {
    pub fn new(
        log_matrix: &CsrMatrix<f64>,
        read_weights: Vec<f64>,
        taxons: Vec<String>,
        taxon_weights: Vec<f64>,
        median_genome_len: f64,
        read_support: usize,
        p_unknown_penalty_ref: f64, 
    ) -> Self {
        println!("Step 3: Converting Matrix to Linear Space for MCMC...");
        
        let mut linear_matrix = log_matrix.clone();
        for val in linear_matrix.values_mut() {
            *val = val.exp(); 
        }

        let total_reads: f64 = read_weights.iter().sum();
        let s = read_support as f64;
        let n = total_reads;
        let g = median_genome_len;
        let p_unk = p_unknown_penalty_ref; 

        let l_null = n * p_unk.ln();
        let w_unk = (n - s) / n;
        let w_sp = s / n;
        
        let prob_match = (p_unk * w_unk) + ((1.0 / g) * w_sp);
        let prob_no_match = p_unk * w_unk;

        let l_one = (s * prob_match.ln()) + ((n - s) * prob_no_match.ln());
        let lpenalty = l_null - l_one;

        println!("Calculated L-Penalty: {:.4}", lpenalty);

        Self {
            matrix: linear_matrix,
            read_weights,
            taxons,
            taxon_weights,
            median_genome_len,
            read_support,
            lpenalty,
        }
    }
}

pub fn run_mini_em(
    ctx: &McmcContext,
    species_set: &HashSet<usize>,
    initial_abundances: &HashMap<usize, f64>,
    start_unk_prob: f64,
    iterations: usize,
) -> (f64, HashMap<usize, f64>, f64) {
    
    let num_reads = ctx.matrix.nrows();
    let active_indices: Vec<usize> = species_set.iter().cloned().collect();
    let num_active = active_indices.len();
    
    let mut abundances: Vec<f64> = active_indices.iter()
        .map(|idx| *initial_abundances.get(idx).unwrap_or(&0.0))
        .collect();
    
    let mut unk_abundance = 0.01;
    let sum: f64 = abundances.iter().sum::<f64>() + unk_abundance;
    for x in &mut abundances { *x /= sum; }
    unk_abundance /= sum;

    let mut curr_unk_prob = start_unk_prob;
    let mut current_log_likelihood = -f64::INFINITY;
    let mut next_abundances = vec![0.0; num_active];
    
    let mut next_unk_abundance;

    for iter in 0..iterations {
        next_abundances.fill(0.0);
        next_unk_abundance = 0.0;
        current_log_likelihood = 0.0;
        
        let mut valid_aligns = if iter > 1 { Vec::with_capacity(num_reads / 10) } else { Vec::new() };

        for (row_idx, row_vec) in ctx.matrix.row_iter().enumerate() {
            let read_weight = ctx.read_weights[row_idx];
            let mut denom_known = 0.0;
            
            for (col_idx, &p_linear) in row_vec.col_indices().iter().zip(row_vec.values()) {
                if let Some(pos) = active_indices.iter().position(|&id| id == *col_idx) {
                    denom_known += p_linear * abundances[pos];
                }
            }
            
            let term_unknown = curr_unk_prob * unk_abundance;
            let total_denom = denom_known + term_unknown;
            let safe_denom = total_denom + 1e-300; 

            current_log_likelihood += read_weight * safe_denom.ln();

            if iter > 1 && denom_known > 1e-300 {
                valid_aligns.push(denom_known);
            }

            let factor = read_weight / safe_denom;
            
            for (col_idx, &p_linear) in row_vec.col_indices().iter().zip(row_vec.values()) {
                if let Some(pos) = active_indices.iter().position(|&id| id == *col_idx) {
                    next_abundances[pos] += factor * p_linear * abundances[pos];
                }
            }
            next_unk_abundance += factor * term_unknown;
        }

        if iter > 1 && valid_aligns.len() > 10 {
            let mid = valid_aligns.len() / 2;
            valid_aligns.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            let median_prob = valid_aligns[mid];

            let proposed_floor = median_prob * 1e-12;
            curr_unk_prob = (0.8 * curr_unk_prob) + (0.2 * proposed_floor);

            if curr_unk_prob < 1e-300 { curr_unk_prob = 1e-300; }
            if curr_unk_prob > 1e-5 { curr_unk_prob = 1e-5; }
        }

        let total_weight: f64 = next_abundances.iter().sum::<f64>() + next_unk_abundance;
        if total_weight > 0.0 {
            for x in &mut next_abundances { *x /= total_weight; }
            next_unk_abundance /= total_weight;
        } else {
            let n = (num_active + 1) as f64;
            next_abundances.fill(1.0/n);
            next_unk_abundance = 1.0/n;
        }

        abundances = next_abundances.clone();
        unk_abundance = next_unk_abundance;
    }

    let mut final_map = HashMap::new();
    for (i, &idx) in active_indices.iter().enumerate() {
        final_map.insert(idx, abundances[i]);
    }

    (current_log_likelihood, final_map, curr_unk_prob)
}

// ================================================================================================
// PART 2: MOVE LOGIC
// ================================================================================================

pub struct MoveProbs {
    pub add: f64,
    pub remove: f64,
    pub swap: f64,
}

#[derive(Clone, Debug)]
pub enum MoveType {
    Add(usize),          
    Remove(usize),       
    Swap(usize, usize),
    None,
}

pub struct McmcLogic;

impl McmcLogic {
    pub fn get_move_probs(num_present_species: usize, num_total_species: usize) -> MoveProbs {
        let can_add = num_present_species < num_total_species;
        let can_remove = num_present_species > 0;

        if !can_remove {
            MoveProbs { add: 1.0, remove: 0.0, swap: 0.0 }
        } else if !can_add {
            MoveProbs { add: 0.0, remove: 1.0, swap: 0.0 }
        } else {
            MoveProbs { add: 0.4, remove: 0.4, swap: 0.2 }
        }
    }

    pub fn pick_add(
        ctx: &McmcContext, 
        current_set: &HashSet<usize>, 
        rng: &mut impl Rng
    ) -> Option<(usize, f64)> {
        let mut candidates = Vec::new();
        let mut weights = Vec::new();
        let mut total_weight = 0.0;

        for id in 0..ctx.matrix.ncols() {
            if !current_set.contains(&id) {
                let w = ctx.taxon_weights[id];
                candidates.push(id);
                weights.push(w);
                total_weight += w;
            }
        }

        if candidates.is_empty() { return None; }
        if total_weight == 0.0 { return None; }

        let dist = WeightedIndex::new(&weights).ok()?;
        let idx = dist.sample(rng);
        Some((candidates[idx], weights[idx] / total_weight))
    }

    pub fn pick_remove(
        current_abundances: &HashMap<usize, f64>,
        rng: &mut impl Rng
    ) -> Option<(usize, f64)> {
        if current_abundances.is_empty() { return None; }

        let mut candidates = Vec::with_capacity(current_abundances.len());
        let mut raw_inv_weights = Vec::with_capacity(current_abundances.len());

        for (&id, &abund) in current_abundances {
            candidates.push(id);
            let inv = 1.0 / (abund + 1e-300);
            raw_inv_weights.push(inv);
        }

        let mut sorted_w = raw_inv_weights.clone();
        sorted_w.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted_w.len();
        let p20 = sorted_w[((len as f64 * 0.2).floor() as usize).clamp(0, len - 1)];
        let p80 = sorted_w[((len as f64 * 0.8).floor() as usize).clamp(0, len - 1)];

        let mut final_weights = Vec::with_capacity(len);
        let mut total_weight = 0.0;

        for &w in &raw_inv_weights {
            let clamped = w.clamp(p20, p80);
            final_weights.push(clamped);
            total_weight += clamped;
        }

        if total_weight == 0.0 { return None; }
        let dist = WeightedIndex::new(&final_weights).ok()?;
        let idx = dist.sample(rng);
        Some((candidates[idx], final_weights[idx] / total_weight))
    }

    pub fn get_pick_prob(
        id: usize,
        move_type: &str, 
        ctx: &McmcContext,
        target_set: &HashSet<usize>,       
        target_abundances: Option<&HashMap<usize, f64>>
    ) -> f64 {
        if move_type == "add" {
            let mut total_weight = 0.0;
            let mut my_weight = 0.0;
            for candidate_id in 0..ctx.matrix.ncols() {
                if !target_set.contains(&candidate_id) {
                    let w = ctx.taxon_weights[candidate_id];
                    total_weight += w;
                    if candidate_id == id { my_weight = w; }
                }
            }
            if total_weight == 0.0 { return 0.0; }
            my_weight / total_weight

        } else {
            let abund_map = target_abundances.expect("Need abundances for remove prob");
            let mut raw_inv_weights = Vec::new();
            let mut my_raw = 0.0;
            
            for (&sp_id, &abund) in abund_map {
                let inv = 1.0 / (abund + 1e-300);
                raw_inv_weights.push(inv);
                if sp_id == id { my_raw = inv; }
            }
            
            let mut sorted_w = raw_inv_weights.clone();
            sorted_w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let len = sorted_w.len();
            
            let p20 = sorted_w[((len as f64 * 0.2).floor() as usize).clamp(0, len - 1)];
            let p80 = sorted_w[((len as f64 * 0.8).floor() as usize).clamp(0, len - 1)];

            let mut total_clamped = 0.0;
            let my_clamped = my_raw.clamp(p20, p80);
            
            for w in raw_inv_weights {
                total_clamped += w.clamp(p20, p80);
            }
            if total_clamped == 0.0 { return 0.0; }
            my_clamped / total_clamped
        }
    }
}

// ================================================================================================
// PART 3: ORCHESTRATOR
// ================================================================================================

pub fn run_chain_step(
    ctx: &McmcContext,
    state: &mut ChainState,
    rng: &mut impl Rng,
    current_iter: usize, // ADDED: Current iteration number
    em_iterations: usize,
    verbose: bool,
) {
    let num_total = ctx.matrix.ncols();
    let num_present = state.species_set.len();
    
    let move_probs = McmcLogic::get_move_probs(num_present, num_total);
    let rand_val: f64 = rng.random();
    
    let move_type = if rand_val < move_probs.add {
        if let Some((id, _)) = McmcLogic::pick_add(ctx, &state.species_set, rng) {
            MoveType::Add(id)
        } else { MoveType::None }
    } else if rand_val < (move_probs.add + move_probs.remove) {
        if let Some((id, _)) = McmcLogic::pick_remove(&state.abundances, rng) {
            MoveType::Remove(id)
        } else { MoveType::None }
    } else {
        if let Some((rem_id, _)) = McmcLogic::pick_remove(&state.abundances, rng) {
            let mut temp_set = state.species_set.clone();
            temp_set.remove(&rem_id);
            if let Some((add_id, _)) = McmcLogic::pick_add(ctx, &temp_set, rng) {
                MoveType::Swap(rem_id, add_id)
            } else { MoveType::None }
        } else { MoveType::None }
    };

    if let MoveType::None = move_type {
        state.history.push(ChainRecord {
            iter: current_iter, 
            log_likelihood: state.current_log_likelihood * state.temperature,
            move_type: "None".to_string(),
        });
        return; 
    }

    state.moves_attempted += 1;

    let mut next_set = state.species_set.clone();
    match move_type {
        MoveType::Add(id) => { next_set.insert(id); },
        MoveType::Remove(id) => { next_set.remove(&id); },
        MoveType::Swap(rem, add) => { next_set.remove(&rem); next_set.insert(add); },
        _ => {}
    }

    let n_species = next_set.len();
    let mut init_abund = HashMap::new();
    
    if n_species > 0 {
        let gamma = Gamma::new(1.0, 1.0).unwrap(); 
        let mut samples = Vec::with_capacity(n_species);
        let mut sum = 0.0;
        
        for _ in 0..n_species {
            let x: f64 = gamma.sample(rng);
            samples.push(x);
            sum += x;
        }
        
        for (i, &id) in next_set.iter().enumerate() {
            init_abund.insert(id, samples[i] / sum);
        }
    }

    let (new_log_l, new_abundances, new_unk_prob) = run_mini_em(
        ctx, 
        &next_set, 
        &init_abund, 
        state.current_unk_prob, 
        em_iterations
    );

    let estimator_new = new_log_l + (next_set.len() as f64 * ctx.lpenalty);
    let current_penalized = state.current_log_likelihood; 
    let new_penalized = estimator_new;

    let log_ratio_data = (new_penalized - current_penalized) * state.temperature;

    let move_probs_new = McmcLogic::get_move_probs(next_set.len(), ctx.matrix.ncols());
    
    let (log_q_fwd, log_q_rev) = match move_type {
        MoveType::Add(id) => {
            let pick_add = McmcLogic::get_pick_prob(id, "add", ctx, &state.species_set, None);
            let pick_rem = McmcLogic::get_pick_prob(id, "remove", ctx, &next_set, Some(&new_abundances));
            
            let fwd = move_probs.add.ln() + pick_add.ln();
            let rev = move_probs_new.remove.ln() + pick_rem.ln();
            (fwd, rev)
        },
        MoveType::Remove(id) => {
            let pick_rem = McmcLogic::get_pick_prob(id, "remove", ctx, &state.species_set, Some(&state.abundances));
            let pick_add = McmcLogic::get_pick_prob(id, "add", ctx, &next_set, None);
            
            let fwd = move_probs.remove.ln() + pick_rem.ln();
            let rev = move_probs_new.add.ln() + pick_add.ln();
            (fwd, rev)
        },
        MoveType::Swap(rem, add) => {
            let pick_rem_fwd = McmcLogic::get_pick_prob(rem, "remove", ctx, &state.species_set, Some(&state.abundances));
            let pick_add_fwd = McmcLogic::get_pick_prob(add, "add", ctx, &state.species_set, None); 
            
            let pick_rem_rev = McmcLogic::get_pick_prob(add, "remove", ctx, &next_set, Some(&new_abundances));
            let pick_add_rev = McmcLogic::get_pick_prob(rem, "add", ctx, &next_set, None);

            let fwd = move_probs.swap.ln() + pick_rem_fwd.ln() + pick_add_fwd.ln();
            let rev = move_probs_new.swap.ln() + pick_rem_rev.ln() + pick_add_rev.ln();
            (fwd, rev)
        },
        _ => (0.0, 0.0),
    };

    let total_log_ratio = log_ratio_data + log_q_rev - log_q_fwd;

    let accept = if total_log_ratio >= 0.0 {
        true
    } else {
        let r: f64 = rng.random();
        r < total_log_ratio.exp()
    };

    if accept {
        state.moves_accepted += 1;
        state.species_set = next_set;
        state.abundances = new_abundances;
        state.current_unk_prob = new_unk_prob;
        state.current_log_likelihood = new_penalized;
        
        if verbose {
            let cid = state.id;
            match move_type {
                MoveType::Add(id) => println!("Chain {} ACCEPTED Add: {}", cid, ctx.taxons[id]),
                MoveType::Remove(id) => println!("Chain {} ACCEPTED Remove: {}", cid, ctx.taxons[id]),
                MoveType::Swap(rem, add) => println!("Chain {} ACCEPTED Swap: {} -> {}", cid, ctx.taxons[rem], ctx.taxons[add]),
                _ => {}
            }
        }

        let m_str = match move_type {
            MoveType::Add(id) => format!("Add({})", id),
            MoveType::Remove(id) => format!("Remove({})", id),
            MoveType::Swap(r, a) => format!("Swap({}->{})", r, a),
            _ => "None".to_string(),
        };
        state.history.push(ChainRecord {
            iter: current_iter,
            log_likelihood: new_penalized * state.temperature,
            move_type: m_str,
        });
    } else {
        state.history.push(ChainRecord {
            iter: current_iter,
            log_likelihood: state.current_log_likelihood * state.temperature,
            move_type: "Reject".to_string(),
        });
    }
}

pub fn run_mcmc_parallel(
    ctx: Arc<McmcContext>,
    initial_states: Vec<ChainState>,
    total_iter: usize,
    exchange_interval: usize,
    verbose: bool,
) -> Result<Vec<ChainState>> {
    let num_chains = initial_states.len();
    
    let shared_chains: Arc<Vec<RwLock<ChainState>>> = Arc::new(
        initial_states.into_iter().map(|s| RwLock::new(s)).collect()
    );

    let barrier = Arc::new(Barrier::new(num_chains));
    let mut handles = vec![];

    for t_id in 0..num_chains {
        let chains_ref = shared_chains.clone();
        let bar_ref = barrier.clone();
        let ctx_ref = ctx.clone();
        
        let handle = thread::spawn(move || {
            let mut rng = rand::rng(); 
            let em_iter = 10; 
            
            let num_blocks = total_iter / exchange_interval;
            
            for block in 0..num_blocks {
                let current_iter_base = block * exchange_interval;

                // 1. Run MCMC Steps
                for i in 0..exchange_interval {
                    let mut state_guard = chains_ref[t_id].write().unwrap();
                    let iter_idx = current_iter_base + i;
                    run_chain_step(&ctx_ref, &mut *state_guard, &mut rng, iter_idx, em_iter, verbose);
                }

                bar_ref.wait();

                // 2. Thread 0 performs swaps & logging
                if t_id == 0 {
                    let current_iter = (block + 1) * exchange_interval;
                    if verbose && (current_iter % 50 == 0 || current_iter == total_iter) {
                        let c0 = chains_ref[0].read().unwrap();
                        println!("Iter: {} / {} | C1 Floor: {:.2e}", current_iter, total_iter, c0.current_unk_prob);
                    }

                    let odd_flag = block % 2; 
                    let start_idx = if odd_flag == 1 { 1 } else { 0 };
                    
                    let mut c = start_idx;
                    while c + 1 < num_chains {
                        let mut chain_a = chains_ref[c].write().unwrap();
                        let mut chain_b = chains_ref[c+1].write().unwrap();
                        
                        chain_a.swaps_attempted += 1;
                        chain_b.swaps_attempted += 1;

                        let l1 = chain_a.current_log_likelihood;
                        let l2 = chain_b.current_log_likelihood;
                        let t1 = chain_a.temperature;
                        let t2 = chain_b.temperature;
                        
                        let log_ratio = (l2 - l1) * (t1 - t2);
                        
                        if log_ratio >= 0.0 || rng.random::<f64>() < log_ratio.exp() {
                            std::mem::swap(&mut chain_a.species_set, &mut chain_b.species_set);
                            std::mem::swap(&mut chain_a.abundances, &mut chain_b.abundances);
                            std::mem::swap(&mut chain_a.current_unk_prob, &mut chain_b.current_unk_prob);
                            std::mem::swap(&mut chain_a.current_log_likelihood, &mut chain_b.current_log_likelihood);
                            
                            chain_a.swaps_accepted += 1;
                            chain_b.swaps_accepted += 1;

                            // LOG THE SWAP (Fix for confusion)
                            let swap_rec_a = ChainRecord {
                                iter: current_iter,
                                log_likelihood: chain_a.current_log_likelihood,
                                move_type: format!("Swapped from Chain {}", c+1),
                            };
                            chain_a.history.push(swap_rec_a);

                            let swap_rec_b = ChainRecord {
                                iter: current_iter,
                                log_likelihood: chain_b.current_log_likelihood,
                                move_type: format!("Swapped from Chain {}", c),
                            };
                            chain_b.history.push(swap_rec_b);
                        }
                        c += 2;
                    }
                }
                bar_ref.wait();
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
    
    let result: Vec<ChainState> = shared_chains.iter()
        .map(|lock| lock.read().unwrap().clone())
        .collect();

    Ok(result)
}