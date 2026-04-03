use std::io::Write;
use std::sync::{Mutex, OnceLock};

static LOG_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

const DEBUG_LOG_PATH: &str = "/Users/roger.barlow/temp/char/.cursor/debug.log";

pub fn append_ndjson_line(value: &serde_json::Value) {
    let lock = LOG_LOCK.get_or_init(|| Mutex::new(()));
    let Ok(_guard) = lock.lock() else {
        return;
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(DEBUG_LOG_PATH)
    {
        let _ = writeln!(f, "{}", value);
    }
}
