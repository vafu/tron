mod page;
mod server;
mod state;

use anyhow::Result;

pub use state::{DiagnosticsHandle, DiagnosticsSnapshot};

pub fn spawn(port: u16) -> Result<DiagnosticsHandle> {
    server::spawn(port)
}

pub fn spawn_available(preferred_port: u16) -> Result<DiagnosticsHandle> {
    server::spawn_available(preferred_port)
}
