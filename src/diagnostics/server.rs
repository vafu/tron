use super::page;
use super::state::DiagnosticsHandle;
use anyhow::Result;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

pub fn spawn(port: u16) -> Result<DiagnosticsHandle> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    spawn_bound(port, listener)
}

pub fn spawn_available(preferred_port: u16) -> Result<DiagnosticsHandle> {
    let last_port = preferred_port.saturating_add(20);
    let mut last_error = None;
    for port in preferred_port..=last_port {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(listener) => {
                if port != preferred_port {
                    eprintln!("diagnostics: port {preferred_port} busy; using {port}");
                }
                return spawn_bound(port, listener);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                last_error = Some(e);
            }
            Err(e) => return Err(e.into()),
        }
    }

    Err(last_error
        .map(anyhow::Error::from)
        .unwrap_or_else(|| anyhow::anyhow!("no diagnostics port candidates available")))
}

fn spawn_bound(port: u16, listener: TcpListener) -> Result<DiagnosticsHandle> {
    let handle = DiagnosticsHandle::new();
    let server_handle = handle.clone();
    eprintln!("diagnostics: http://127.0.0.1:{port}/");

    thread::Builder::new()
        .name("diagnostics-http".into())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(mut stream) => {
                        if let Err(e) = serve(&mut stream, &server_handle) {
                            eprintln!("diagnostics: request failed: {e:#}");
                        }
                    }
                    Err(e) => eprintln!("diagnostics: accept failed: {e:#}"),
                }
            }
        })?;

    Ok(handle)
}

fn serve(stream: &mut TcpStream, handle: &DiagnosticsHandle) -> Result<()> {
    let mut buf = [0u8; 2048];
    let n = stream.read(&mut buf)?;
    let request = String::from_utf8_lossy(&buf[..n]);
    let Some(first_line) = request.lines().next() else {
        write_response(stream, 400, "text/plain", b"bad request")?;
        return Ok(());
    };
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();
    if method != "GET" {
        write_response(stream, 405, "text/plain", b"method not allowed")?;
        return Ok(());
    }

    match path {
        "/" | "/index.html" => write_response(
            stream,
            200,
            "text/html; charset=utf-8",
            page::INDEX_HTML.as_bytes(),
        )?,
        "/state.json" => {
            let body = handle.snapshot_json();
            write_response(
                stream,
                200,
                "application/json; charset=utf-8",
                body.as_bytes(),
            )?;
        }
        "/health" => write_response(stream, 200, "text/plain", b"ok")?,
        _ => write_response(stream, 404, "text/plain", b"not found")?,
    }
    Ok(())
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {status_text}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Cache-Control: no-cache\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}
