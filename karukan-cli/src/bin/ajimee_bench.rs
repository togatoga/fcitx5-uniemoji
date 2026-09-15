//! AJIMEE-Bench evaluation tool for karukan-engine
//!
//! Evaluates the Rust llama.cpp integration against AJIMEE-Bench,
//! producing metrics comparable to the Python `jinen-evaluate` tool.

use anyhow::{Context, Result};
use clap::Parser;
use karukan_engine::kana::normalize_nfkc;
use karukan_engine::kanji::{build_jinen_prompt, clean_model_output};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// AJIMEE-Bench evaluation for karukan-engine
#[derive(Parser)]
#[command(name = "ajimee-bench")]
struct Cli {
    /// Path to evaluation_items.json
    bench_path: PathBuf,

    /// Model variant id (e.g. jinen-v2-xsmall-q5, jinen-v2-small-q5)
    #[arg(long, default_value = "jinen-v2-xsmall-q5")]
    model: String,

    /// Direct GGUF file path (overrides --model)
    #[arg(long)]
    gguf: Option<PathBuf>,

    /// Path to tokenizer.json (required when using --gguf)
    #[arg(long)]
    tokenizer_json: Option<PathBuf>,

    /// Save detailed results to JSON
    #[arg(long)]
    output: Option<PathBuf>,

    /// Disable left context usage
    #[arg(long)]
    no_context: bool,

    /// Show only summary
    #[arg(long)]
    quiet: bool,

    /// Typing simulation: convert every growing prefix of each reading
    /// (first 30 items) and report per-keystroke latency. This is the
    /// live-conversion access pattern, where KV prefix reuse applies;
    /// the independent-items evaluation cannot show it.
    #[arg(long)]
    typing: bool,

    /// Context window size
    #[arg(long, default_value_t = 512)]
    n_ctx: u32,
}

/// A single AJIMEE-Bench evaluation item
#[derive(Debug, Deserialize)]
struct BenchItem {
    input: String,
    context_text: Option<String>,
    expected_output: Vec<String>,
}

/// Result for a single evaluation item
#[derive(Debug, Serialize)]
struct ItemResult {
    input: String,
    context: String,
    prediction: String,
    expected: Vec<String>,
    exact_match: bool,
    min_cer: f64,
    nfkc_prediction: String,
    nfkc_expected: Vec<String>,
    nfkc_exact_match: bool,
    nfkc_min_cer: f64,
}

/// Overall evaluation metrics
#[derive(Debug, Serialize)]
struct Metrics {
    num_examples: usize,
    exact_match_rate: f64,
    avg_min_cer: f64,
    nfkc_exact_match_rate: f64,
    nfkc_avg_min_cer: f64,
    results: Vec<ItemResult>,
}

/// Truncate text at the first occurrence of any stop token
fn truncate_at_stop_tokens(text: &str) -> &str {
    let stops = ["\u{ee00}", "\u{ee02}", "\n"];
    let mut end = text.len();
    for s in &stops {
        if let Some(p) = text.find(s) {
            end = end.min(p);
        }
    }
    &text[..end]
}

/// Levenshtein distance at the character level (Wagner-Fischer)
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Calculate Character Error Rate
fn calculate_cer(hypothesis: &str, reference: &str) -> f64 {
    let ref_len = reference.chars().count();
    if ref_len == 0 {
        return if hypothesis.is_empty() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    levenshtein_distance(hypothesis, reference) as f64 / ref_len as f64
}

/// Calculate minimum CER across multiple references
fn calculate_min_cer(hypothesis: &str, references: &[String]) -> f64 {
    if references.is_empty() {
        return f64::INFINITY;
    }
    references
        .iter()
        .map(|r| calculate_cer(hypothesis, r))
        .fold(f64::INFINITY, f64::min)
}

/// Convert every growing prefix of each reading, timing each call: the
/// per-keystroke cost of live conversion.
fn run_typing_sim(
    model: &karukan_engine::kanji::LlamaCppModel,
    items: &[BenchItem],
    no_context: bool,
    eos: Option<i32>,
) -> Result<()> {
    use std::time::Instant;
    const ITEMS: usize = 30;

    let mut lat_ms: Vec<f64> = Vec::new();
    for item in items.iter().filter(|i| !i.input.is_empty()).take(ITEMS) {
        let context = if no_context {
            ""
        } else {
            item.context_text.as_deref().unwrap_or("")
        };
        let chars: Vec<char> = item.input.chars().collect();
        for i in 1..=chars.len() {
            let prefix: String = chars[..i].iter().collect();
            let start = Instant::now();
            let prompt = build_jinen_prompt(&prefix, context);
            let tokens = model.tokenize(&prompt)?;
            model.generate(&tokens, 100, eos)?;
            lat_ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
    }

    lat_ms.sort_by(|a, b| a.total_cmp(b));
    let n = lat_ms.len();
    let total: f64 = lat_ms.iter().sum();
    let median = lat_ms[n / 2];
    let p95 = lat_ms[((n as f64 * 0.95) as usize).min(n - 1)];
    println!("Typing simulation ({ITEMS} items, {n} keystrokes)");
    println!(
        "total {:.2}s  mean {:.1}ms  median {:.1}ms  p95 {:.1}ms per keystroke",
        total / 1000.0,
        total / n as f64,
        median,
        p95
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load model
    let model = karukan_cli::load_llama_model(
        cli.gguf.as_deref(),
        cli.tokenizer_json.as_deref(),
        &cli.model,
        cli.n_ctx,
    )?;

    let eos = Some(model.eos_token_id().0);

    // Load benchmark
    eprintln!("Loading benchmark from {}...", cli.bench_path.display());
    let data = std::fs::read_to_string(&cli.bench_path)
        .with_context(|| format!("Failed to read {}", cli.bench_path.display()))?;
    let items: Vec<BenchItem> =
        serde_json::from_str(&data).context("Failed to parse evaluation_items.json")?;
    eprintln!("Loaded {} examples", items.len());

    if cli.typing {
        return run_typing_sim(&model, &items, cli.no_context, eos);
    }

    let mut results: Vec<ItemResult> = Vec::with_capacity(items.len());
    let mut exact_matches = 0usize;
    let mut total_cer = 0.0f64;
    let mut nfkc_exact_matches = 0usize;
    let mut nfkc_total_cer = 0.0f64;

    for (idx, item) in items.iter().enumerate() {
        if item.input.is_empty() {
            continue;
        }

        // The input is already katakana in AJIMEE-Bench
        let reading = &item.input;

        let context = if cli.no_context {
            ""
        } else {
            item.context_text.as_deref().unwrap_or("")
        };

        // Build prompt and generate
        let prompt = build_jinen_prompt(reading, context);
        let tokens = model
            .tokenize(&prompt)
            .with_context(|| format!("Failed to tokenize example {}", idx + 1))?;

        let output_tokens = model.generate(&tokens, 100, eos)?;
        let generated = &output_tokens[tokens.len()..];
        let text = model.decode(generated, true)?;

        // Post-process: clean special tokens, truncate at stop tokens
        let cleaned = clean_model_output(&text);
        let prediction = truncate_at_stop_tokens(&cleaned).trim().to_string();

        // Evaluate (original)
        let exact_match = item.expected_output.contains(&prediction);
        let min_cer = calculate_min_cer(&prediction, &item.expected_output);

        // Evaluate (NFKC-normalized)
        let nfkc_prediction = normalize_nfkc(&prediction);
        let nfkc_expected: Vec<String> = item
            .expected_output
            .iter()
            .map(|e| normalize_nfkc(e))
            .collect();
        let nfkc_exact_match = nfkc_expected.contains(&nfkc_prediction);
        let nfkc_min_cer = calculate_min_cer(&nfkc_prediction, &nfkc_expected);

        if exact_match {
            exact_matches += 1;
        }
        total_cer += min_cer;
        if nfkc_exact_match {
            nfkc_exact_matches += 1;
        }
        nfkc_total_cer += nfkc_min_cer;

        let n = results.len() + 1;

        if !cli.quiet {
            let status = if exact_match { "OK" } else { "NG" };
            let nfkc_status = if nfkc_exact_match { "OK" } else { "NG" };
            let ng = n - exact_matches;
            let nfkc_ng = n - nfkc_exact_matches;
            println!(
                "\n[{}] {} (OK:{} NG:{}) | NFKC: {} (OK:{} NG:{})",
                n, status, exact_matches, ng, nfkc_status, nfkc_exact_matches, nfkc_ng
            );
            println!("  Input (katakana): {}", reading);
            if !context.is_empty() {
                println!("  Context:  {}", context);
            }
            println!("  Output:   {}", prediction);
            println!("  Expected: {:?}", item.expected_output);
            if prediction != nfkc_prediction || item.expected_output != nfkc_expected {
                println!("  Output   (NFKC): {}", nfkc_prediction);
                println!("  Expected (NFKC): {:?}", nfkc_expected);
            }
        }

        results.push(ItemResult {
            input: reading.clone(),
            context: context.to_string(),
            prediction,
            expected: item.expected_output.clone(),
            exact_match,
            min_cer,
            nfkc_prediction,
            nfkc_expected,
            nfkc_exact_match,
            nfkc_min_cer,
        });
    }

    let num_examples = results.len();
    let exact_match_rate = if num_examples > 0 {
        exact_matches as f64 / num_examples as f64
    } else {
        0.0
    };
    let avg_min_cer = if num_examples > 0 {
        total_cer / num_examples as f64
    } else {
        0.0
    };
    let nfkc_exact_match_rate = if num_examples > 0 {
        nfkc_exact_matches as f64 / num_examples as f64
    } else {
        0.0
    };
    let nfkc_avg_min_cer = if num_examples > 0 {
        nfkc_total_cer / num_examples as f64
    } else {
        0.0
    };

    // Print summary
    println!();
    println!("{}", "=".repeat(50));
    println!("Evaluation Results");
    println!("{}", "=".repeat(50));
    println!("Number of examples: {}", num_examples);
    println!("Exact match rate:   {:.2}%", exact_match_rate * 100.0);
    println!("Avg min CER:        {:.4}", avg_min_cer);
    println!("{}", "-".repeat(50));
    println!(
        "Exact match rate (NFKC): {:.2}%",
        nfkc_exact_match_rate * 100.0
    );
    println!("Avg min CER      (NFKC): {:.4}", nfkc_avg_min_cer);
    println!("{}", "=".repeat(50));

    // Save detailed results if requested
    if let Some(output_path) = &cli.output {
        let metrics = Metrics {
            num_examples,
            exact_match_rate,
            avg_min_cer,
            nfkc_exact_match_rate,
            nfkc_avg_min_cer,
            results,
        };
        let json = serde_json::to_string_pretty(&metrics)?;
        std::fs::write(output_path, &json)
            .with_context(|| format!("Failed to write {}", output_path.display()))?;
        eprintln!("Detailed results saved to {}", output_path.display());
    }

    Ok(())
}
