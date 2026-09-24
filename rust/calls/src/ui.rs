//! egui panels: transport, genome sliders, call view, XY pad, library, archive.

use bevy_egui::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2, pos2, vec2};

use crate::State;
use crate::audio::Source;
use crate::gesture::{ALPHA_MAX, ALPHA_MIN, ALPHA_ONSET, BETA_MAX, BETA_MIN};
use crate::params::{GROUPS, N_PARAMS, SPEC, default_phys};
use crate::spectro::{F_HI, F_LO, magma};

const ALPHA_COL: Color32 = Color32::from_rgb(235, 90, 100);
const BETA_COL: Color32 = Color32::from_rgb(60, 200, 210);

pub fn draw(ctx: &egui::Context, s: &mut State) {
    shortcuts(ctx, s);
    top_bar(ctx, s);
    egui::SidePanel::left("genome").resizable(false).exact_width(430.0).show(ctx, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| genome_panel(ui, s));
    });
    egui::SidePanel::right("library").resizable(false).exact_width(340.0).show(ctx, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            library_panel(ui, s);
            ui.separator();
            archive_panel(ui, s);
        });
    });
    egui::CentralPanel::default().show(ctx, |ui| {
        let h = ui.available_height();
        call_view(ui, s, (h * 0.5).max(200.0));
        ui.add_space(6.0);
        ui.horizontal_top(|ui| {
            let side = ui.available_height().min(ui.available_width() * 0.45).max(160.0);
            xy_pad(ui, s, side);
            live_view(ui, s, side);
        });
    });
    // Keep the playhead and live spectrogram moving.
    ctx.request_repaint();
}

fn shortcuts(ctx: &egui::Context, s: &mut State) {
    if ctx.wants_keyboard_input() {
        return;
    }
    let (space, m, r, undo, redo) = ctx.input(|i| {
        let cmd = i.modifiers.command;
        (
            i.key_pressed(egui::Key::Space),
            i.key_pressed(egui::Key::M),
            i.key_pressed(egui::Key::R) && !cmd,
            cmd && !i.modifiers.shift && i.key_pressed(egui::Key::Z),
            cmd && (i.key_pressed(egui::Key::Y) || (i.modifiers.shift && i.key_pressed(egui::Key::Z))),
        )
    });
    if space {
        if s.audio.status.active.load(std::sync::atomic::Ordering::Relaxed) { s.stop() } else { s.play() }
    }
    if m {
        s.mutate();
        s.play();
    }
    if r {
        s.randomize(0..N_PARAMS);
        s.play();
    }
    if undo {
        s.undo();
    }
    if redo {
        s.redo();
    }
}

fn top_bar(ctx: &egui::Context, s: &mut State) {
    egui::TopBottomPanel::top("transport").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let active = s.audio.status.active.load(std::sync::atomic::Ordering::Relaxed);
            if ui.button(if active { "Stop" } else { "Play" }).on_hover_text("space").clicked() {
                if active { s.stop() } else { s.play() }
            }
            let mut c = s.audio.control.lock().unwrap();
            ui.checkbox(&mut c.looping, "loop");
            ui.separator();
            ui.label("source");
            ui.selectable_value(&mut c.source, Source::Call, "CPG call");
            ui.selectable_value(&mut c.source, Source::Pad, "XY pad");
            ui.separator();
            ui.add(egui::Slider::new(&mut c.volume, 0.0..=1.0).text("volume"));
            drop(c);
            ui.separator();
            if ui.add(egui::Slider::new(&mut s.duration, 1.0..=10.0).step_by(0.25).text("duration s"))
                .on_hover_text("window length; the phrase occupies the middle third")
                .changed()
            {
                s.dirty = true;
            }
            if ui.add(egui::DragValue::new(&mut s.seed).prefix("noise seed ")).changed() {
                s.dirty = true;
            }
            ui.separator();
            if ui.add_enabled(s.history_pos > 0, egui::Button::new("Undo")).on_hover_text("undo (ctrl+z)").clicked() {
                s.undo();
            }
            if ui.add_enabled(s.history_pos + 1 < s.history.len(), egui::Button::new("Redo")).on_hover_text("redo (ctrl+shift+z)").clicked() {
                s.redo();
            }
            ui.separator();
            if let Some(r) = &s.rendered {
                ui.weak(format!("render {:.0} ms · {:.0} Hz out", r.ms, s.audio.sample_rate));
            }
            ui.weak(&s.status);
        });
    });
}

fn genome_panel(ui: &mut egui::Ui, s: &mut State) {
    ui.horizontal_wrapped(|ui| {
        if ui.button("Mindlin default").clicked() {
            s.set_phys(default_phys());
            s.play();
        }
        if ui.button("Randomize").on_hover_text("unlocked params (r)").clicked() {
            s.randomize(0..N_PARAMS);
            s.play();
        }
        if ui.button("Mutate").on_hover_text("gaussian step on unlocked params (m)").clicked() {
            s.mutate();
            s.play();
        }
        ui.add(egui::Slider::new(&mut s.sigma, 0.01..=0.5).logarithmic(true).text("σ"));
    });
    ui.separator();
    ui.spacing_mut().slider_width = 230.0;

    for (title, lo, hi) in GROUPS {
        egui::CollapsingHeader::new(title).default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.small_button("randomize").on_hover_text("randomize this group").clicked() {
                    s.randomize(lo..hi);
                    s.play();
                }
                let all = s.locked[lo..hi].iter().all(|&l| l);
                if ui.small_button(if all { "unlock all" } else { "lock all" }).clicked() {
                    s.locked[lo..hi].iter_mut().for_each(|l| *l = !all);
                }
                if ui.small_button("reset").clicked() {
                    let mut p = s.phys;
                    p[lo..hi].copy_from_slice(&default_phys()[lo..hi]);
                    s.set_phys(p);
                }
            });
            egui::Grid::new(title).num_columns(3).spacing([6.0, 2.0]).show(ui, |ui| {
                for i in lo..hi {
                    let spec = &SPEC[i];
                    ui.checkbox(&mut s.locked[i], "").on_hover_text("lock (skip in randomize / mutate)");
                    ui.label(spec.name).on_hover_text(spec.help);
                    let mut v = s.phys[i];
                    let resp = ui.add(
                        egui::Slider::new(&mut v, spec.lo..=spec.hi)
                            .logarithmic(spec.log && spec.lo > 0.0)
                            .max_decimals(if spec.hi - spec.lo > 100.0 { 0 } else { 3 }),
                    );
                    let resp = resp.on_hover_text(spec.help);
                    if resp.changed() {
                        s.phys[i] = v;
                        s.changed();
                    }
                    if resp.drag_stopped() || (resp.changed() && !resp.dragged()) {
                        s.commit();
                    }
                    if resp.double_clicked() {
                        s.phys[i] = spec.default;
                        s.set_phys(s.phys);
                    }
                    ui.end_row();
                }
            });
        });
    }
}

fn call_view(ui: &mut egui::Ui, s: &mut State, height: f32) {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::click());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 0.0, Color32::BLACK);
    let Some(r) = &s.rendered else {
        p.text(rect.center(), Align2::CENTER_CENTER, "rendering…", FontId::proportional(16.0), Color32::GRAY);
        return;
    };
    let plot = Rect::from_min_max(rect.min + vec2(44.0, 4.0), rect.max - vec2(4.0, 18.0));
    if let Some(t) = &s.call_tex {
        p.image(t.id(), plot, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    }
    freq_axis(&p, plot);
    let secs = r.wave.len() as f64 / crate::RENDER_SR;
    for k in 0..=(secs.floor() as usize) {
        let x = plot.left() + plot.width() * (k as f64 / secs) as f32;
        p.text(pos2(x, plot.bottom() + 2.0), Align2::CENTER_TOP, format!("{k}s"), FontId::monospace(10.0), Color32::GRAY);
    }

    // Motor gesture over the spectrogram, each in its own range.
    let n = r.gesture.alpha.len();
    let w = plot.width().max(1.0) as usize;
    let trace = |xs: &[f32], lo: f64, hi: f64| -> Vec<Pos2> {
        (0..w)
            .map(|c| {
                let v = xs[(c * n / w).min(n - 1)] as f64;
                let u = ((v - lo) / (hi - lo)) as f32;
                pos2(plot.left() + c as f32, plot.bottom() - u * plot.height())
            })
            .collect()
    };
    let onset_y = plot.bottom() - ((ALPHA_ONSET - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)) as f32 * plot.height();
    p.line_segment([pos2(plot.left(), onset_y), pos2(plot.right(), onset_y)], stroke(0.5, ALPHA_COL.gamma_multiply(0.5)));
    p.line(trace(&r.gesture.alpha, ALPHA_MIN, ALPHA_MAX), stroke(1.2, ALPHA_COL.gamma_multiply(0.8)));
    p.line(trace(&r.gesture.beta, BETA_MIN, BETA_MAX), stroke(1.2, BETA_COL.gamma_multiply(0.8)));
    p.text(plot.left_top() + vec2(6.0, 4.0), Align2::LEFT_TOP, "α pressure", FontId::proportional(12.0), ALPHA_COL);
    p.text(plot.left_top() + vec2(80.0, 4.0), Align2::LEFT_TOP, "β tension", FontId::proportional(12.0), BETA_COL);

    // Playhead, from the gesture the audio thread is walking.
    let c = s.audio.control.lock().unwrap();
    let dev_n = c.gesture.alpha.len().max(1);
    let playing = c.source == Source::Call && s.audio.status.active.load(std::sync::atomic::Ordering::Relaxed);
    drop(c);
    if playing {
        let pos = s.audio.status.position.load(std::sync::atomic::Ordering::Relaxed);
        let x = plot.left() + plot.width() * (pos as f32 / dev_n as f32);
        p.line_segment([pos2(x, plot.top()), pos2(x, plot.bottom())], stroke(1.5, Color32::WHITE));
    }
    if resp.clicked() {
        s.play();
    }
    resp.on_hover_text("click to play from the top");
}

fn freq_axis(p: &egui::Painter, plot: Rect) {
    for f in [200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0] {
        let u = ((f / F_LO as f32).ln() / (F_HI / F_LO).ln() as f32).clamp(0.0, 1.0);
        let y = plot.bottom() - u * plot.height();
        let label = if f >= 1000.0 { format!("{}k", f / 1000.0) } else { format!("{f}") };
        p.text(pos2(plot.left() - 4.0, y), Align2::RIGHT_CENTER, label, FontId::monospace(10.0), Color32::GRAY);
        p.line_segment([pos2(plot.left() - 2.0, y), pos2(plot.left(), y)], stroke(1.0, Color32::GRAY));
    }
}

fn xy_pad(ui: &mut egui::Ui, s: &mut State, side: f32) {
    ui.vertical(|ui| {
        ui.horizontal(|ui| {
            ui.strong("XY pad");
            ui.checkbox(&mut s.pad_hold, "hold").on_hover_text("keep blowing after you let go");
        });
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(side - 24.0), Sense::click_and_drag());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 4.0, Color32::from_gray(18));
        p.rect_stroke(rect, 4.0, stroke(1.0, Color32::from_gray(60)), StrokeKind::Inside);

        let to_screen = |a: f64, b: f64| {
            pos2(
                rect.left() + ((a - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)) as f32 * rect.width(),
                rect.bottom() - ((b - BETA_MIN) / (BETA_MAX - BETA_MIN)) as f32 * rect.height(),
            )
        };
        let onset = to_screen(ALPHA_ONSET, 0.0).x;
        p.rect_filled(Rect::from_min_max(rect.min, pos2(onset, rect.bottom())), 4.0, Color32::from_black_alpha(90));
        p.text(pos2(onset + 4.0, rect.top() + 4.0), Align2::LEFT_TOP, "phonation onset", FontId::proportional(10.0), Color32::GRAY);
        p.text(rect.center_bottom() - vec2(0.0, 4.0), Align2::CENTER_BOTTOM, "α pressure", FontId::proportional(11.0), ALPHA_COL);
        p.text(rect.left_center() + vec2(4.0, 0.0), Align2::LEFT_CENTER, "β tension", FontId::proportional(11.0), BETA_COL);

        // The current call's path through the (α, β) plane.
        if let Some(r) = &s.rendered {
            let n = r.gesture.alpha.len();
            let step = (n / 1500).max(1);
            let pts: Vec<Pos2> = (0..n).step_by(step).map(|k| to_screen(r.gesture.alpha[k] as f64, r.gesture.beta[k] as f64)).collect();
            p.line(pts, stroke(1.0, Color32::from_rgba_unmultiplied(255, 200, 120, 90)));
        }

        let mut c = s.audio.control.lock().unwrap();
        let pressed = resp.is_pointer_button_down_on();
        if pressed {
            if let Some(pt) = resp.interact_pointer_pos() {
                let a = ALPHA_MIN + ((pt.x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64 * (ALPHA_MAX - ALPHA_MIN);
                let b = BETA_MIN + ((rect.bottom() - pt.y) / rect.height()).clamp(0.0, 1.0) as f64 * (BETA_MAX - BETA_MIN);
                c.pad = (a, b);
                c.source = Source::Pad;
            }
        }
        c.pad_on = c.source == Source::Pad && (pressed || s.pad_hold);
        if c.source == Source::Call && c.playing && s.audio.status.active.load(std::sync::atomic::Ordering::Relaxed) {
            let pos = s.audio.status.position.load(std::sync::atomic::Ordering::Relaxed);
            if let (Some(&a), Some(&b)) = (c.gesture.alpha.get(pos), c.gesture.beta.get(pos)) {
                p.circle_filled(to_screen(a as f64, b as f64), 5.0, Color32::from_rgb(255, 200, 120));
            }
        }
        if c.source == Source::Pad {
            let col = if c.pad_on { Color32::WHITE } else { Color32::GRAY };
            p.circle_stroke(to_screen(c.pad.0, c.pad.1), 7.0, stroke(2.0, col));
            ui.weak(format!("α {:+.3}   β {:.3}", c.pad.0, c.pad.1));
        } else {
            ui.weak("drag to play the instrument by hand");
        }
    });
}

fn live_view(ui: &mut egui::Ui, s: &mut State, side: f32) {
    ui.vertical(|ui| {
        ui.strong("live output");
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), side - 24.0), Sense::hover());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, Color32::BLACK);
        let plot = Rect::from_min_max(rect.min + vec2(44.0, 0.0), rect.max);
        if let Some(t) = &s.live_tex {
            p.image(t.id(), plot, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        }
        freq_axis(&p, plot);
    });
}

fn library_panel(ui: &mut egui::Ui, s: &mut State) {
    ui.heading("Sounds");
    ui.horizontal(|ui| {
        ui.label("name");
        ui.text_edit_singleline(&mut s.preset_name);
    });
    ui.add(egui::TextEdit::multiline(&mut s.preset_notes).hint_text("notes").desired_rows(2).desired_width(f32::INFINITY));
    ui.horizontal(|ui| {
        if ui.button("Save").on_hover_text(format!("{}/<name>.json", s.preset_dir.display())).clicked() {
            s.save_preset();
        }
        if ui.add_enabled(s.rendered.is_some(), egui::Button::new("Export wav")).on_hover_text(format!("{}/<name>.wav (48 kHz)", s.export_dir.display())).clicked() {
            s.export_wav();
        }
        if ui.button("Rescan").on_hover_text("rescan presets folder").clicked() {
            s.refresh_presets();
        }
    });
    ui.add(egui::TextEdit::singleline(&mut s.preset_filter).hint_text("filter"));
    let filter = s.preset_filter.to_lowercase();
    let mut load = None;
    let mut delete = None;
    for (k, e) in s.presets.iter().enumerate() {
        if !filter.is_empty() && !e.preset.name.to_lowercase().contains(&filter) && !e.preset.notes.to_lowercase().contains(&filter) {
            continue;
        }
        ui.horizontal(|ui| {
            let current = e.preset.to_phys() == s.phys;
            let r = ui.selectable_label(current, &e.preset.name);
            let r = if e.preset.notes.is_empty() { r } else { r.on_hover_text(&e.preset.notes) };
            if r.clicked() {
                load = Some(k);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let armed = s.confirm_delete.as_ref() == Some(&e.path);
                if ui.small_button(if armed { "sure?" } else { "delete" }).clicked() {
                    delete = Some(k);
                }
            });
        });
    }
    if let Some(k) = load {
        s.load_preset(k);
    }
    if let Some(k) = delete {
        let path = s.presets[k].path.clone();
        if s.confirm_delete.as_ref() == Some(&path) {
            let _ = std::fs::remove_file(&path);
            s.confirm_delete = None;
            s.status = format!("deleted {}", path.display());
            s.refresh_presets();
        } else {
            s.confirm_delete = Some(path);
        }
    }
    if s.presets.is_empty() {
        ui.weak("no saved sounds yet");
    }
}

fn archive_panel(ui: &mut egui::Ui, s: &mut State) {
    ui.heading("MAP-Elites archive");
    let current = s.archive.as_ref().map(|a| short(&a.path)).unwrap_or_else(|| "choose a run…".into());
    let mut pick = None;
    egui::ComboBox::from_id_salt("archive").selected_text(current).width(ui.available_width() - 10.0).show_ui(ui, |ui| {
        for path in &s.archive_paths {
            if ui.selectable_label(false, short(path)).clicked() {
                pick = Some(path.clone());
            }
        }
    });
    if let Some(path) = pick {
        s.load_archive(path);
    }
    let Some(a) = &s.archive else {
        if s.archive_paths.is_empty() {
            ui.weak("no archives found under python/calls/outputs");
        }
        return;
    };
    let (lo, hi) = a.fitness_range();
    let side = ui.available_width().min(320.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(side), Sense::click());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 0.0, Color32::from_gray(15));
    let cell = side / a.res as f32;
    let cell_rect = |i: usize, j: usize| {
        let min = pos2(rect.left() + i as f32 * cell, rect.bottom() - (j + 1) as f32 * cell);
        Rect::from_min_size(min, Vec2::splat(cell))
    };
    for i in 0..a.res {
        for j in 0..a.res {
            let f = a.fitness(i, j);
            if f.is_finite() {
                let t = if hi > lo { ((f - lo) / (hi - lo)) as f32 } else { 1.0 };
                p.rect_filled(cell_rect(i, j).shrink(0.5), 0.0, magma(0.25 + 0.75 * t));
            }
        }
    }
    if let Some((i, j)) = s.archive_cell {
        p.rect_stroke(cell_rect(i, j), 0.0, stroke(2.0, Color32::WHITE), StrokeKind::Outside);
    }
    let hovered = resp.hover_pos().map(|pt| {
        let i = ((pt.x - rect.left()) / cell).floor().clamp(0.0, a.res as f32 - 1.0) as usize;
        let j = ((rect.bottom() - pt.y) / cell).floor().clamp(0.0, a.res as f32 - 1.0) as usize;
        (i, j)
    });
    ui.weak("x: BirdNET PC1 · y: PC2 · colour: fitness");
    if let Some((i, j)) = hovered {
        let f = a.fitness(i, j);
        ui.label(if f.is_finite() { format!("cell ({i}, {j})  fitness {f:.3}") } else { format!("cell ({i}, {j})  empty") });
        if resp.clicked() && f.is_finite() {
            s.load_cell(i, j);
        }
    }
}

fn short(path: &std::path::Path) -> String {
    path.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn stroke(width: f32, color: Color32) -> Stroke {
    Stroke::new(width, color)
}
