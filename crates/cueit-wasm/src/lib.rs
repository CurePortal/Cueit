//! WebAssembly bindings for CueIt.
//!
//! The browser decodes audio with the Web Audio API (`AudioContext.decodeAudioData`)
//! and hands the mono `Float32Array` here. Everything else — beat/tempo/key
//! detection, structure, cue inference and the Camelot mapping — is the exact
//! same `cueit-core` code the desktop app and CLI use.
//!
//! Build with:
//! ```bash
//! wasm-pack build crates/cueit-wasm --target web --no-default-features
//! ```

use wasm_bindgen::prelude::*;

use cueit_core::analysis::analyze_pcm;
use cueit_core::cue::infer_cues;
use cueit_core::types::AnalysisResult;

/// Run the full DSP + key + structure pipeline on decoded mono PCM.
///
/// `samples` is the mono channel at `sample_rate` Hz. Returns a JS object
/// matching the `AnalysisResult` shape used across the app.
#[wasm_bindgen(js_name = analyze)]
pub fn analyze_samples(samples: &[f32], sample_rate: u32, path: &str) -> Result<JsValue, JsError> {
    let result = analyze_pcm(path, samples, sample_rate, None);
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
}

/// Re-run cue inference on an existing analysis (no re-DSP).
#[wasm_bindgen(js_name = inferCues)]
pub fn infer_cues_js(analysis: JsValue, use_downbeats: bool) -> Result<JsValue, JsError> {
    let a: AnalysisResult = serde_wasm_bindgen::from_value(analysis)
        .map_err(|e| JsError::new(&format!("bad analysis object: {e}")))?;
    let cues = infer_cues(&a, use_downbeats);
    serde_wasm_bindgen::to_value(&cues.cues).map_err(|e| JsError::new(&e.to_string()))
}

/// Camelot wheel code for a pitch class (0 = C) and major/minor mode.
#[wasm_bindgen(js_name = camelot)]
pub fn camelot_js(pc: u32, major: bool) -> String {
    cueit_core::analysis::camelot(pc as usize, major).to_string()
}
