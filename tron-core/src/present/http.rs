use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::thread;
use tron_api::Presenter;

#[derive(Clone)]
pub struct HttpMetadataPresenter {
    addr: SocketAddr,
    state: Arc<Mutex<String>>,
}

impl HttpMetadataPresenter {
    pub fn bind(addr: impl ToSocketAddrs) -> Result<Self> {
        let listener = TcpListener::bind(addr).context("bind metadata HTTP listener")?;
        spawn_bound(listener)
    }

    pub fn bind_available(addr: impl ToSocketAddrs, additional_ports: u16) -> Result<Self> {
        let mut addrs = addr.to_socket_addrs().context("resolve metadata address")?;
        let addr = addrs
            .next()
            .ok_or_else(|| anyhow::anyhow!("metadata address did not resolve"))?;
        let last_port = addr.port().saturating_add(additional_ports);
        let mut last_error = None;

        for port in addr.port()..=last_port {
            match TcpListener::bind(SocketAddr::new(addr.ip(), port)) {
                Ok(listener) => {
                    if port != addr.port() {
                        eprintln!("metadata-http: port {} busy; using {}", addr.port(), port);
                    }
                    return spawn_bound(listener);
                }
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::AddrInUse {
                        last_error = Some(err);
                        continue;
                    }
                    return Err(err).context("bind metadata HTTP listener");
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::AddrInUse,
                "no metadata HTTP port candidates available",
            )
        }))
        .context("bind metadata HTTP listener")
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }
}

fn spawn_bound(listener: TcpListener) -> Result<HttpMetadataPresenter> {
    let addr = listener
        .local_addr()
        .context("read metadata HTTP address")?;
    let state = Arc::new(Mutex::new("{}".to_owned()));
    let server_state = state.clone();

    thread::Builder::new()
        .name("tron-http-metadata".into())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(mut stream) => {
                        if let Err(err) = serve(&mut stream, &server_state) {
                            eprintln!("metadata-http: request failed: {err:#}");
                        }
                    }
                    Err(err) => eprintln!("metadata-http: accept failed: {err:#}"),
                }
            }
        })
        .context("spawn metadata HTTP thread")?;

    Ok(HttpMetadataPresenter { addr, state })
}

impl<V> Presenter<V> for HttpMetadataPresenter
where
    V: serde::Serialize,
{
    fn present(&mut self, view: V) -> Result<()> {
        let body = serde_json::to_string(&view).context("serialize metadata HTTP view")?;
        *self.state.lock().expect("metadata state lock poisoned") = body;
        Ok(())
    }
}

fn serve(stream: &mut TcpStream, state: &Arc<Mutex<String>>) -> Result<()> {
    let mut buf = [0u8; 2048];
    let n = stream
        .read(&mut buf)
        .context("read metadata HTTP request")?;
    let request = String::from_utf8_lossy(&buf[..n]);
    let Some(first_line) = request.lines().next() else {
        write_response(stream, 400, "text/plain; charset=utf-8", b"bad request")?;
        return Ok(());
    };
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();

    if method != "GET" {
        write_response(
            stream,
            405,
            "text/plain; charset=utf-8",
            b"method not allowed",
        )?;
        return Ok(());
    }

    match path {
        "/" | "/index.html" => write_response(
            stream,
            200,
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes(),
        )?,
        "/metadata" | "/metadata.json" => {
            let body = state.lock().expect("metadata state lock poisoned").clone();
            write_response(
                stream,
                200,
                "application/json; charset=utf-8",
                body.as_bytes(),
            )?;
        }
        "/health" => write_response(stream, 200, "text/plain; charset=utf-8", b"ok")?,
        _ => write_response(stream, 404, "text/plain; charset=utf-8", b"not found")?,
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
         Cache-Control: no-store\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    )
    .context("write metadata HTTP headers")?;
    stream.write_all(body).context("write metadata HTTP body")?;
    stream.flush().context("flush metadata HTTP response")?;
    Ok(())
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tron metadata</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071013;
      --panel: #0d181d;
      --line: #203640;
      --text: #d8eef3;
      --muted: #82a3ad;
      --accent: #61d9ee;
      --warn: #f2c75c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    main {
      width: min(1320px, calc(100vw - 32px));
      margin: 20px auto;
      display: grid;
      gap: 14px;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
    }
    h1 {
      margin: 0;
      color: var(--accent);
      font-size: 18px;
      font-weight: 700;
    }
    #status {
      color: var(--muted);
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      padding: 14px;
      min-width: 0;
    }
    h2 {
      margin: 0 0 12px;
      color: var(--accent);
      font-size: 13px;
      text-transform: uppercase;
    }
    dl {
      display: grid;
      grid-template-columns: minmax(120px, 0.7fr) minmax(0, 1fr);
      gap: 8px 16px;
      margin: 0;
    }
    dt { color: var(--muted); }
    dd {
      margin: 0;
      overflow-wrap: anywhere;
      text-align: right;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
    }
    .warn { color: var(--warn); }
    @media (max-width: 780px) {
      .grid { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>tron metadata</h1>
      <div id="status">connecting</div>
    </header>
    <div class="grid">
      <section>
        <h2>RGB</h2>
        <dl id="rgb"></dl>
      </section>
      <section>
        <h2>IR</h2>
        <dl id="ir"></dl>
      </section>
    </div>
    <section>
      <h2>Timing</h2>
      <dl id="timing"></dl>
    </section>
    <section>
      <h2>Raw JSON</h2>
      <pre id="json">{}</pre>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById('status');
    const jsonEl = document.getElementById('json');
    const rgbEl = document.getElementById('rgb');
    const irEl = document.getElementById('ir');
    const timingEl = document.getElementById('timing');

    function valueText(value, unit = '') {
      if (value === null || value === undefined) return 'null';
      if (typeof value === 'number') {
        const text = Number.isInteger(value) ? String(value) : value.toFixed(3);
        return unit ? `${text} ${unit}` : text;
      }
      return String(value);
    }

    function renderDl(el, rows) {
      el.replaceChildren(...rows.flatMap(([key, value, unit]) => {
        const dt = document.createElement('dt');
        const dd = document.createElement('dd');
        dt.textContent = key;
        dd.textContent = valueText(value, unit);
        if (value === null || value === undefined) dd.className = 'warn';
        return [dt, dd];
      }));
    }

    function renderCamera(el, camera) {
      camera ||= {};
      renderDl(el, [
        ['sensor', camera.sensor],
        ['fps', camera.fps],
        ['frame delta', camera.frame_delta_us, 'us'],
        ['age', camera.age_us, 'us'],
        ['frame id', camera.frame_id],
        ['sequence', camera.sequence],
        ['camera timestamp', camera.camera_monotonic_us, 'us'],
      ]);
    }

    async function refresh() {
      const started = performance.now();
      try {
        const response = await fetch('/metadata', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        renderCamera(rgbEl, data.rgb);
        renderCamera(irEl, data.ir);
        renderDl(timingEl, [
          ['rgb-ir delta', data.rgb_ir_delta_us, 'us'],
          ['http fetch', performance.now() - started, 'ms'],
        ]);
        jsonEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = `updated ${new Date().toLocaleTimeString()}`;
        statusEl.className = '';
      } catch (error) {
        statusEl.textContent = `error: ${error.message}`;
        statusEl.className = 'warn';
      }
    }

    refresh();
    setInterval(refresh, 250);
  </script>
</body>
</html>
"#;
