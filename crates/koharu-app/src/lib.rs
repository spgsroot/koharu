//! Koharu's Tauri-managed application state, commands, and lifecycle.

mod app;
mod automation;
mod commands;
mod history;
mod timings;

pub use app::run;
pub use commands::bindings;
