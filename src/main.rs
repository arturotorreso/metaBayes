use clap::Parser;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

mod step1_parser;
mod step2_reduce;
mod step3_mcmc;
mod step4_inference;

#[derive(Parser)]
#[command(name = "MetaMix-RS")]
struct Cli {
    /// BAM file sorted by read name (eg: output of samtools sort -n)
    #[arg(short, long)]
    input: String,
    /// Output prefix
    #[arg(short, long, default_value = "metamix_out")]
    output: String,
    /// Output posterior distribution of read counts per species
    #[arg(long, default_value_t = false)]
    output_posterior: bool,
    #[arg(short, long)]
    threads: Option<usize>,
    /// Path to NCBI names.dmp file for scientific name mapping
    #[arg(long)]
    taxonomy_names: Option<String>,

    #[arg(long, default_value_t = false)]
    verbose: bool,
    // --- STEP 2 ARGS ---
    /// Minimum aligned reads to consider a species valid for EM
    #[arg(long, default_value_t = 1)]
    em_read_cutoff: usize,
    /// Maximum EM iterations
    #[arg(long, default_value_t = 1000)]
    em_iter: usize,

    // --- STEP 3 ARGS ---
    /// Number of MCMC chains
    #[arg(long, default_value_t = 12)]
    chains: usize,
    /// Total MCMC iterations per chain
    #[arg(long, default_value_t = 1000)]
    mcmc_iter: usize,
    /// Exchange interval for Parallel Tempering
    #[arg(long, default_value_t = 1)]
    exchange_interval: usize,
    /// Read support threshold for Penalty calculation
    #[arg(long, default_value_t = 30)]
    read_support: usize,
    /// Median genome length (default: auto-detected from BAM header)
    #[arg(long)]
    median_genome_len: Option<f64>,

    // --- STEP 4 ARGS ---
    /// Number of iterations for the final Gibbs sampler
    #[arg(long, default_value_t = 100)]
    gibbs_iter: usize,
    /// Number of burn-in iterations for the final Gibbs sampler
    #[arg(long, default_value_t = 20)]
    gibbs_burnin: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()?;
    }

    println!("\n=== STEP 1: BAM PARSING ===");
    let (entries, num_reads, parser, read_names, detected_median_len) = step1_parser::process_bam(&cli.input)?;

    let final_median_len = cli.median_genome_len.unwrap_or(detected_median_len);

    println!("\n=== STEP 2: EM DIMENSION REDUCTION ===");
    
    let step2_result = step2_reduce::run_em_reduction(
        entries, 
        num_reads, 
        parser, 
        cli.em_read_cutoff, 
        cli.em_iter,
        cli.verbose
    )?;

    println!("\n=== STEP 3: MCMC PARALLEL TEMPERING ===");

    let read_weights = vec![1.0; num_reads];

    let ctx = Arc::new(step3_mcmc::McmcContext::new(
        &step2_result.matrix,
        read_weights,
        step2_result.reduced_taxons.clone(),    
        step2_result.reduced_abundances.clone(),
        final_median_len,
        cli.read_support,
        1e-20, 
    ));

    let mut initial_states = Vec::with_capacity(cli.chains);
    let k_temp = 0.001;
    let a_temp = 1.5;
    let mut prev_temp = 1.0;

    for i in 0..cli.chains {
        let temp = if i == 0 { 1.0 } else {
            let base: f64 = prev_temp - k_temp;
            if base < 0.0 { 0.0 } else { base.powf(a_temp) }
        };
        prev_temp = temp;

        if cli.verbose {
            println!("Initializing Chain {} (Temp: {:.4})", i, temp);
        }

        let species_set = HashSet::new();
        let abundances = HashMap::new(); 

        initial_states.push(step3_mcmc::ChainState {
            id: i,
            temperature: temp,
            species_set,
            abundances,
            current_unk_prob: 1e-300, 
            current_log_likelihood: -1e10, 
            moves_attempted: 0,
            moves_accepted: 0,
            swaps_attempted: 0,
            swaps_accepted: 0,
            history: Vec::with_capacity(cli.mcmc_iter),
        });
    }

    if !cli.verbose {
        println!("Initialized {} chains.", cli.chains);
    }

    let final_chains = step3_mcmc::run_mcmc_parallel(
        ctx.clone(),
        initial_states,
        cli.mcmc_iter,
        cli.exchange_interval,
        cli.verbose 
    )?;

    println!("MCMC Completed.");

    println!("\n=== STEP 4: INFERENCE ===");
    let taxonomy_map = if let Some(path) = &cli.taxonomy_names {
        Some(step4_inference::load_taxonomy_names(path)?)
    } else {
        None
    };

    step4_inference::run_inference(
        &ctx,
        &final_chains,
        0.1, // Burnin ratio for trace
        &cli.output,
        taxonomy_map.as_ref(),
        &read_names, 
        cli.output_posterior,
        cli.gibbs_iter,
        cli.gibbs_burnin,
    )?;

    println!("\nPipeline Completed Successfully.");
    Ok(())
}