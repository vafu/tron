use super::*;

impl Gfx {
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.main_view.fit(&self.queue, size);
        self.masked_view.fit(&self.queue, size);
        self.mask_view.fit(&self.queue, size);
        self.depth = DepthTexture::new(&self.device, size);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let log_start = self
            .render_timing
            .last_log
            .get_or_insert_with(Instant::now)
            .to_owned();
        let t_lock = Instant::now();
        // Pull latest data
        let (rgb_opt, hand, mask_opt, pointer) = {
            let _span = tracing::debug_span!("gfx.lock_inputs").entered();
            (
                self.rgb_src.lock().unwrap().clone(),
                self.hand_src.lock().unwrap().clone(),
                self.mask_src.lock().unwrap().clone(),
                *self.pointer_src.lock().unwrap(),
            )
        };
        self.render_timing.lock_us += t_lock.elapsed().as_micros() as u64;

        let t_upload = Instant::now();
        {
            let _span = tracing::debug_span!("gfx.upload").entered();
            // Main pane: raw RGB camera (skeleton drawn on top later).
            if let Some(f) = &rgb_opt {
                if f.seq != self.main_view.last_seq {
                    self.main_view.upload(&self.queue, &f.data);
                    self.main_view.last_seq = f.seq;
                }
            }

            // Masked-RGB pane: pipeline's debug image (RGB darkened by IR mask).
            // Falls back to raw RGB until the pipeline produces its first frame.
            if let Some(state) = &hand {
                if let Some(dbg) = &state.debug_image {
                    if dbg.seq != self.masked_view.last_seq {
                        self.masked_view.upload(&self.queue, &dbg.data);
                        self.masked_view.last_seq = dbg.seq;
                    }
                }
            } else if let Some(f) = &rgb_opt {
                if f.seq != self.masked_view.last_seq {
                    self.masked_view.upload(&self.queue, &f.data);
                    self.masked_view.last_seq = f.seq;
                }
            }

            // Mask pane: grayscale IR-diff. Texture is RGBA8 so expand R8 → RGBA8.
            if let Some(m) = &mask_opt {
                if m.seq != self.mask_view.last_seq {
                    let needed = (m.width * m.height) as usize * 4;
                    if self.mask_rgba.len() != needed {
                        self.mask_rgba.resize(needed, 255);
                    }
                    expand_to_rgba(m, &mut self.mask_rgba);
                    self.mask_view.upload(&self.queue, &self.mask_rgba);
                    self.mask_view.last_seq = m.seq;
                }
            }
        }
        self.render_timing.upload_us += t_upload.elapsed().as_micros() as u64;

        let t_overlay = Instant::now();
        let mut have_overlay = false;
        let prox;
        {
            let _span = tracing::debug_span!("gfx.overlay_update").entered();
            // Update prox bar fill
            prox = *self.prox_src.lock().unwrap();
            if let Some(p) = prox {
                if p > self.prox_max {
                    self.prox_max = p;
                }
                let norm = (p as f32 / self.prox_max as f32).clamp(0.0, 1.0);
                self.bar_fill
                    .set_rect(&self.queue, (-1.0, -1.0, -1.0 + 2.0 * norm, -0.97));
            }

            // Pull hand state, update skeleton + ROI mesh.
            if let Some(state) = &hand {
                have_overlay = true;
                if self.options.skeleton {
                    let clip = letterbox_rect(
                        self.main_pane,
                        self.main_view.w,
                        self.main_view.h,
                        self.size.width,
                        self.size.height,
                    );
                    self.skeleton.update(
                        &self.queue,
                        Some(&state.landmarks),
                        Some(&state.roi),
                        clip,
                        (self.size.width, self.size.height),
                        state.gesture,
                        state.gesture.map(|g| g.name()),
                    );
                }
            }
        }

        // Keep gesture feedback in the overlay; the title remains for coarse
        // process/sensor status only.
        let ir_mask = if self.controls.ir_mask_enabled() {
            "ir-mask:on"
        } else {
            "ir-mask:off"
        };
        let title = match prox {
            Some(p) => format!("tron — prox: {p} — {ir_mask}"),
            None => format!("tron — {ir_mask}"),
        };
        self.window.set_title(&title);

        if self.options.cube {
            if let Some(pointer) = pointer {
                if pointer.grabbed {
                    let pos = [pointer.position.x, pointer.position.y];
                    if let Some(last) = self.last_grab_pos {
                        let dx = pos[0] - last[0];
                        let dy = pos[1] - last[1];
                        self.cube.rotate(-dx * 8.0, -dy * 8.0);
                    }
                    self.last_grab_pos = Some(pos);
                } else {
                    self.last_grab_pos = None;
                }
            } else {
                self.last_grab_pos = None;
            }
            self.cube
                .update(&self.queue, self.size.width, self.size.height);
        } else {
            self.last_grab_pos = None;
        }
        self.render_timing.overlay_us += t_overlay.elapsed().as_micros() as u64;

        let t_encode = Instant::now();
        let _encode_span = tracing::debug_span!("gfx.encode").entered();
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp-2d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.tex_pipeline);
            for q in [&self.main_view, &self.masked_view, &self.mask_view] {
                rp.set_bind_group(0, &q.bind_group, &[]);
                rp.set_vertex_buffer(0, q.vbuf.slice(..));
                rp.draw(0..6, 0..1);
            }

            rp.set_pipeline(&self.solid_pipeline);
            for q in [&self.bar_bg, &self.bar_fill] {
                rp.set_bind_group(0, &q.bind_group, &[]);
                rp.set_vertex_buffer(0, q.vbuf.slice(..));
                rp.draw(0..6, 0..1);
            }
        }

        if self.options.cube {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp-cube"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.cube.draw(&mut rp);
        }

        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rp-overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if self.options.cube {
                self.cube.draw_overlay(&mut rp);
            }
            if have_overlay && self.options.skeleton {
                self.skeleton.draw(&mut rp);
            }
        }
        self.render_timing.encode_us += t_encode.elapsed().as_micros() as u64;
        drop(_encode_span);
        let t_submit = Instant::now();
        {
            let _span = tracing::debug_span!("gfx.submit").entered();
            self.queue.submit(Some(enc.finish()));
            frame.present();
        }
        self.render_timing.submit_us += t_submit.elapsed().as_micros() as u64;
        self.render_timing.frames += 1;
        self.log_render_timing(log_start);
        Ok(())
    }

    fn log_render_timing(&mut self, log_start: Instant) {
        let elapsed = log_start.elapsed();
        if elapsed < Duration::from_secs(2) {
            return;
        }
        let n = self.render_timing.frames.max(1) as f32;
        tracing::debug!(
            target: "tron::gfx",
            fps = self.render_timing.frames as f32 / elapsed.as_secs_f32(),
            lock_ms = self.render_timing.lock_us as f32 / n / 1000.0,
            upload_ms = self.render_timing.upload_us as f32 / n / 1000.0,
            overlay_ms = self.render_timing.overlay_us as f32 / n / 1000.0,
            encode_ms = self.render_timing.encode_us as f32 / n / 1000.0,
            submit_ms = self.render_timing.submit_us as f32 / n / 1000.0,
            "render timing"
        );
        self.render_timing = RenderTiming {
            last_log: Some(Instant::now()),
            ..Default::default()
        };
    }
}
