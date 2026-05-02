use super::{LABEL_PAD_PX, LABEL_SCALE_PX, MAX_LABEL_CHARS, V};

pub(super) fn push_label(
    out: &mut Vec<V>,
    text: &str,
    top_right: [f32; 2],
    ndcx: f32,
    ndcy: f32,
    alert: f32,
) {
    let text = text.chars().take(MAX_LABEL_CHARS).collect::<Vec<_>>();
    let width_px = text_width_px(text.len());
    let x0 = top_right[0] - (width_px + LABEL_PAD_PX) * ndcx;
    let y0 = top_right[1] - LABEL_PAD_PX * ndcy;

    let mut cursor_px = 0.0;
    for ch in text {
        if let Some(rows) = glyph(ch) {
            for (row, bits) in rows.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) == 0 {
                        continue;
                    }
                    let x = x0 + (cursor_px + col as f32 * LABEL_SCALE_PX) * ndcx;
                    let y = y0 - row as f32 * LABEL_SCALE_PX * ndcy;
                    push_label_cell(out, x, y, LABEL_SCALE_PX, ndcx, ndcy, alert);
                }
            }
        }
        cursor_px += 6.0 * LABEL_SCALE_PX;
    }
}

fn text_width_px(chars: usize) -> f32 {
    if chars == 0 {
        0.0
    } else {
        (chars as f32 * 6.0 - 1.0) * LABEL_SCALE_PX
    }
}

pub(super) fn push_label_cell(
    out: &mut Vec<V>,
    x: f32,
    y: f32,
    size_px: f32,
    ndcx: f32,
    ndcy: f32,
    alert: f32,
) {
    let w = size_px * ndcx;
    let h = size_px * ndcy;
    let tl = [x, y];
    let tr = [x + w, y];
    let bl = [x, y - h];
    let br = [x + w, y - h];
    for pos in [tl, bl, br, tl, br, tr] {
        out.push(V {
            pos,
            uv: [0.0, 0.0],
            kind: 3.0,
            intensity: 1.0,
            alert,
        });
    }
}

fn glyph(ch: char) -> Option<[u8; 7]> {
    Some(match ch.to_ascii_lowercase() {
        'a' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'c' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'e' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'f' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'h' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'i' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
        'k' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'm' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'n' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'o' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'p' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        's' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        't' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'u' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '?' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100,
        ],
        _ => return None,
    })
}
