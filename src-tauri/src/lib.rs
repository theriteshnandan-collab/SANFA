use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;
use notify::{Watcher, RecursiveMode, EventKind};
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::thread;
use serde::Serialize;

#[derive(Clone, Serialize)]
struct LogPayload {
    message: String,
    level: String,
}

fn emit_log(app: &AppHandle, msg: &str, level: &str) {
    let _ = app.emit("app-log", LogPayload {
        message: msg.to_string(),
        level: level.to_string(),
    });
}

/// Core processing function: takes an image, runs the Python sidecar, emits logs
fn process_image(app: AppHandle, input_path: PathBuf, output_path: PathBuf) {
    tauri::async_runtime::spawn(async move {
        let file_name = input_path.file_name().unwrap_or_default().to_string_lossy().to_string();
        emit_log(&app, &format!("Detected new asset: {}", file_name), "info");

        let sidecar_command = match app.shell().sidecar("engine") {
            Ok(cmd) => cmd,
            Err(e) => {
                emit_log(&app, &format!("Sidecar error: {}", e), "error");
                return;
            }
        };

        let (mut rx, mut _child) = match sidecar_command
            .arg(input_path.to_string_lossy().to_string())
            .arg(output_path.to_string_lossy().to_string())
            .spawn()
        {
            Ok(pair) => pair,
            Err(e) => {
                emit_log(&app, &format!("Failed to spawn engine: {}", e), "error");
                return;
            }
        };

        emit_log(&app, "Applying structural noise...", "info");

        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    let text = String::from_utf8_lossy(&line);
                    let clean_text = text.trim();
                    if clean_text.is_empty() { continue; }

                    if clean_text.starts_with("SUCCESS:") {
                        emit_log(&app, "Asset shielded successfully.", "success");
                        let _ = app.emit("asset-shielded", ());
                    } else if clean_text.starts_with("PROGRESS:") || clean_text.starts_with("REPORT:") {
                        // Forward progress and report data directly to frontend
                        emit_log(&app, clean_text, "info");
                    } else if clean_text.starts_with("ENGINE:") {
                        // Engine status messages
                        emit_log(&app, &clean_text[7..], "info");
                    } else {
                        emit_log(&app, clean_text, "info");
                    }
                }
                CommandEvent::Stderr(line) => {
                    let text = String::from_utf8_lossy(&line);
                    let clean_text = text.trim();
                    if !clean_text.is_empty() {
                        emit_log(&app, &format!("Engine error: {}", clean_text), "error");
                    }
                }
                _ => {}
            }
        }
    });
}

#[tauri::command]
fn get_app_paths(app: AppHandle) -> (String, String) {
    let app_dir = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    let raw_dir = app_dir.join("Raw Art");
    let shielded_dir = app_dir.join("Shielded Art");
    (
        raw_dir.to_string_lossy().to_string(),
        shielded_dir.to_string_lossy().to_string()
    )
}

#[tauri::command]
async fn open_folder(folder_type: String, app: AppHandle) -> Result<(), String> {
    let app_dir = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    let path = if folder_type == "raw" {
        app_dir.join("Raw Art")
    } else {
        app_dir.join("Shielded Art")
    };

    std::fs::create_dir_all(&path).ok();

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Called by the frontend when user picks a file via the native dialog.
/// This directly triggers processing — NO file watcher involved, so exactly 1 event.
#[tauri::command]
async fn ingest_file(source_path: String, app: AppHandle) -> Result<(), String> {
    let source = PathBuf::from(&source_path);
    if !source.exists() {
        return Err("Source file does not exist".into());
    }

    let file_name = source.file_name().unwrap_or_default();
    let app_dir = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    let shielded_dir = app_dir.join("Shielded Art");
    std::fs::create_dir_all(&shielded_dir).ok();

    let output_path = shielded_dir.join(file_name);

    emit_log(&app, &format!("Ingested file: {:?}", file_name), "info");

    // Process directly — skip the watcher entirely to avoid duplicate events
    process_image(app.clone(), source, output_path);

    Ok(())
}

/// Background file watcher for users who manually drag files into the Raw Art folder.
/// Uses a deduplication HashMap to ignore duplicate filesystem events from the OS.
fn setup_watcher(app: AppHandle) {
    std::thread::spawn(move || {
        let app_dir = app.path().app_data_dir().unwrap_or_else(|_| PathBuf::from("."));
        let raw_dir = app_dir.join("Raw Art");
        let shielded_dir = app_dir.join("Shielded Art");

        std::fs::create_dir_all(&raw_dir).ok();
        std::fs::create_dir_all(&shielded_dir).ok();

        // Deduplication: track recently seen files with timestamps
        let seen: Arc<Mutex<HashMap<PathBuf, Instant>>> = Arc::new(Mutex::new(HashMap::new()));
        let debounce_window = Duration::from_secs(3);

        let (tx, rx) = channel();
        let mut watcher = notify::recommended_watcher(tx).unwrap();
        watcher.watch(&raw_dir, RecursiveMode::NonRecursive).unwrap();

        emit_log(&app, "File watcher initialized. Standing by...", "info");

        for res in rx {
            match res {
                Ok(event) => {
                    // Only react to Create events to minimize duplicates
                    if let EventKind::Create(_) = event.kind {
                        for path in event.paths {
                            if path.is_file() {
                                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                                    let ext_lower = ext.to_lowercase();
                                    if ext_lower == "png" || ext_lower == "jpg" || ext_lower == "jpeg" {
                                        // Deduplication check
                                        let mut lock = seen.lock().unwrap();
                                        let now = Instant::now();
                                        if let Some(last_seen) = lock.get(&path) {
                                            if now.duration_since(*last_seen) < debounce_window {
                                                continue; // Skip duplicate
                                            }
                                        }
                                        lock.insert(path.clone(), now);
                                        drop(lock);

                                        // Small delay to let the file finish writing
                                        thread::sleep(Duration::from_millis(800));

                                        let file_name = path.file_name().unwrap();
                                        let out_path = shielded_dir.join(file_name);
                                        process_image(app.clone(), path.clone(), out_path);
                                    }
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    emit_log(&app, &format!("Watch error: {:?}", e), "error");
                }
            }
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            setup_watcher(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_app_paths, open_folder, ingest_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
