use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use tron_api::Sink;

#[derive(Clone)]
pub struct HttpJsonSink {
    addr: SocketAddr,
    state: Arc<Mutex<HttpState>>,
}

struct HttpState {
    latest: String,
    subscribers: Vec<mpsc::Sender<String>>,
}

impl HttpJsonSink {
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

fn spawn_bound(listener: TcpListener) -> Result<HttpJsonSink> {
    let addr = listener
        .local_addr()
        .context("read metadata HTTP address")?;
    let state = Arc::new(Mutex::new(HttpState {
        latest: "{}".to_owned(),
        subscribers: Vec::new(),
    }));
    let server_state = state.clone();

    thread::Builder::new()
        .name("tron-http-metadata".into())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let request_state = server_state.clone();
                        thread::spawn(move || {
                            if let Err(err) = serve(stream, &request_state) {
                                eprintln!("metadata-http: request failed: {err:#}");
                            }
                        });
                    }
                    Err(err) => eprintln!("metadata-http: accept failed: {err:#}"),
                }
            }
        })
        .context("spawn metadata HTTP thread")?;

    Ok(HttpJsonSink { addr, state })
}

#[async_trait::async_trait(?Send)]
impl<V> Sink<V> for HttpJsonSink
where
    V: serde::Serialize,
{
    async fn consume<'a>(&'a mut self, view: V) -> Result<()>
    where
        V: 'a,
    {
        let body = serde_json::to_string(&view).context("serialize HTTP JSON view")?;
        let mut state = self.state.lock().expect("metadata state lock poisoned");
        state.latest = body.clone();
        state
            .subscribers
            .retain(|subscriber| subscriber.send(body.clone()).is_ok());
        Ok(())
    }
}

fn serve(mut stream: TcpStream, state: &Arc<Mutex<HttpState>>) -> Result<()> {
    let mut buf = [0u8; 2048];
    let n = stream
        .read(&mut buf)
        .context("read metadata HTTP request")?;
    let request = String::from_utf8_lossy(&buf[..n]);
    let Some(first_line) = request.lines().next() else {
        write_response(
            &mut stream,
            400,
            "text/plain; charset=utf-8",
            b"bad request",
        )?;
        return Ok(());
    };
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();

    if method != "GET" {
        write_response(
            &mut stream,
            405,
            "text/plain; charset=utf-8",
            b"method not allowed",
        )?;
        return Ok(());
    }

    match path {
        "/" | "/index.html" => write_response(
            &mut stream,
            200,
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes(),
        )?,
        "/metadata" | "/metadata.json" => {
            let body = state
                .lock()
                .expect("metadata state lock poisoned")
                .latest
                .clone();
            write_response(
                &mut stream,
                200,
                "application/json; charset=utf-8",
                body.as_bytes(),
            )?;
        }
        "/events" => serve_events(stream, state)?,
        "/health" => write_response(&mut stream, 200, "text/plain; charset=utf-8", b"ok")?,
        _ => write_response(&mut stream, 404, "text/plain; charset=utf-8", b"not found")?,
    }

    Ok(())
}

fn serve_events(mut stream: TcpStream, state: &Arc<Mutex<HttpState>>) -> Result<()> {
    let (tx, rx) = mpsc::channel();
    let initial = {
        let mut state = state.lock().expect("metadata state lock poisoned");
        let initial = state.latest.clone();
        state.subscribers.push(tx);
        initial
    };
    write_event_stream_headers(&mut stream)?;
    write_sse_event(&mut stream, &initial)?;
    for body in rx {
        if write_sse_event(&mut stream, &body).is_err() {
            break;
        }
    }
    Ok(())
}

fn write_event_stream_headers(stream: &mut TcpStream) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Cache-Control: no-store\r\n\
         Content-Type: text/event-stream\r\n\
         Connection: keep-alive\r\n\
         \r\n\
         retry: 1000\r\n\
         \r\n"
    )
    .context("write metadata event stream headers")?;
    stream.flush().context("flush metadata event stream")?;
    Ok(())
}

fn write_sse_event(stream: &mut TcpStream, body: &str) -> Result<()> {
    writeln!(stream, "event: metadata").context("write metadata event name")?;
    for line in body.lines() {
        writeln!(stream, "data: {line}").context("write metadata event body")?;
    }
    writeln!(stream).context("finish metadata event")?;
    stream.flush().context("flush metadata event")
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

    const cameraStats = new Map();

    function frameFor(data, name) {
      const direct = data?.[name];
      if (direct?.meta) return direct;
      if (name === 'ir') return data?.ir_diff?.meta ? data.ir_diff : data?.depth_cue?.meta ? data.depth_cue : direct;
      return direct;
    }

    function cameraRows(name, frame) {
      const meta = frame?.meta || {};
      const timestamp = meta.timestamp || {};
      const previous = cameraStats.get(name);
      const now = performance.now();
      let fps = null;
      let frameDeltaUs = null;

      if (previous && meta.id !== undefined && previous.id !== meta.id) {
        const elapsedMs = now - previous.seenAt;
        if (elapsedMs > 0) {
          fps = 1000 / elapsedMs;
        }
        if (
          timestamp.camera_monotonic_us !== undefined &&
          timestamp.camera_monotonic_us !== null &&
          previous.cameraMonotonicUs !== undefined &&
          previous.cameraMonotonicUs !== null
        ) {
          frameDeltaUs = timestamp.camera_monotonic_us - previous.cameraMonotonicUs;
        }
      }

      if (meta.id !== undefined) {
        cameraStats.set(name, {
          id: meta.id,
          seenAt: now,
          cameraMonotonicUs: timestamp.camera_monotonic_us,
        });
      }

      return [
        ['sensor', meta.sensor],
        ['format', frame?.format],
        ['fps', fps],
        ['frame delta', frameDeltaUs, 'us'],
        ['age', null, 'us'],
        ['frame id', meta.id],
        ['sequence', meta.sequence],
        ['camera timestamp', timestamp.camera_monotonic_us, 'us'],
      ];
    }

    function renderCamera(el, name, frame) {
      renderDl(el, [
        ...cameraRows(name, frame),
      ]);
    }

    let lastEventAt = null;

    function renderMetadata(data) {
      const now = performance.now();
      const eventGap = lastEventAt === null ? null : now - lastEventAt;
      lastEventAt = now;
      renderCamera(rgbEl, 'rgb', frameFor(data, 'rgb'));
      renderCamera(irEl, 'ir', frameFor(data, 'ir'));
      renderDl(timingEl, [
        ['rgb-ir delta', data.rgb_ir_delta_us ?? data.sync_delta_us, 'us'],
        ['event gap', eventGap, 'ms'],
      ]);
      jsonEl.textContent = JSON.stringify(data, null, 2);
      statusEl.textContent = `updated ${new Date().toLocaleTimeString()}`;
      statusEl.className = '';
    }

    function connectEvents() {
      const events = new EventSource('/events');
      events.addEventListener('open', () => {
        statusEl.textContent = 'connected';
        statusEl.className = '';
      });
      events.addEventListener('metadata', (event) => {
        try {
          renderMetadata(JSON.parse(event.data));
        } catch (error) {
          statusEl.textContent = `error: ${error.message}`;
          statusEl.className = 'warn';
        }
      });
      events.addEventListener('error', () => {
        statusEl.textContent = 'reconnecting';
        statusEl.className = 'warn';
      });
    }

    async function fetchOnce() {
      try {
        const response = await fetch('/metadata', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        renderMetadata(data);
      } catch (error) {
        statusEl.textContent = `error: ${error.message}`;
        statusEl.className = 'warn';
      }
    }

    connectEvents();
    fetchOnce();
  </script>
</body>
</html>
"#;
