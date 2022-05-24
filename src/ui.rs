use std::num::NonZeroU32;

use bevy::{
    diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin},
    math::DVec2,
    prelude::*,
};
use bevy_egui::{
    egui::{self, panel::Side, Rgba},
    EguiContext,
};
use massi::{
    cranelift::{compile, ModuleError},
    parser::{Error, Full},
    Expr, Identifier,
};

use crate::physics::{Bounds, Gravity, LinkConstraint, ObjectPos, PhysSettings, PointConstraint};

#[cfg(feature = "math")]
enum ExprRes {
    Expr(Expr),
    Error(Vec<Error<Full>>),
}

#[cfg(feature = "math")]
impl Default for ExprRes {
    fn default() -> Self {
        Self::Expr(Expr::empty())
    }
}

#[derive(Default)]
pub struct State {
    #[cfg(feature = "math")]
    expr_x: ExprRes,
    #[cfg(feature = "math")]
    x_compile_err: Option<ModuleError>,
    #[cfg(feature = "math")]
    expr_y: ExprRes,
    #[cfg(feature = "math")]
    y_compile_err: Option<ModuleError>,
}

pub fn ui(
    mut commands: Commands,
    mut objects: Query<(Entity, &mut ObjectPos)>,
    links: Query<(Entity, &LinkConstraint<Entity>)>,
    points: Query<(Entity, &PointConstraint<Entity>)>,
    mut egui_context: ResMut<EguiContext>,
    mut settings: ResMut<PhysSettings>,
    diagnostics: Res<Diagnostics>,
    mut state: Local<State>,
) {
    #[cfg(feature = "tracy")]
    profiling::scope!("ui system");
    fn scalar(ui: &mut egui::Ui, label: &str, v: &mut f64) {
        ui.horizontal(|ui| {
            ui.label(label);
            ui.add(egui::DragValue::new(v));
        });
    }

    fn vector(ui: &mut egui::Ui, label: &str, v: &mut DVec2) {
        ui.horizontal(|ui| {
            ui.label(label);
            ui.add(egui::DragValue::new(&mut v.x));
            ui.add(egui::DragValue::new(&mut v.y));
        });
    }

    egui::SidePanel::new(Side::Left, "settings").show(egui_context.ctx_mut(), |ui| {
        ui.heading("Settings");

        let mut curr = settings.bounds.as_str();
        egui::ComboBox::from_label("Bounds")
            .selected_text(curr)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut curr, "Circle", "Circle");
                ui.selectable_value(&mut curr, "Rect", "Rect");
                ui.selectable_value(&mut curr, "None", "None");
            });

        settings.bounds = Bounds::from_str(curr, settings.bounds.clone());

        match &mut settings.bounds {
            Bounds::Circle(radius) => {
                scalar(ui, "Radius", radius);
                *radius = radius.max(0.0);
            }
            Bounds::Rect(min, max) => {
                vector(ui, "Min", min);
                vector(ui, "Max", max);
                *min = min.min(*max);
                *max = max.max(*min);
            }
            Bounds::None => {}
        }

        let mut curr = settings.gravity.as_str();
        egui::ComboBox::from_label("Gravity")
            .selected_text(curr)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut curr, "Dir", "Dir");
                ui.selectable_value(&mut curr, "Towards", "Towards");
                #[cfg(feature = "math")]
                ui.selectable_value(&mut curr, "Vector Field", "Vector Field");
                ui.selectable_value(&mut curr, "None", "None");
            });
        settings.gravity = Gravity::from_str(curr, settings.gravity.clone());

        match &mut settings.gravity {
            Gravity::Dir(dir) => {
                vector(ui, "Dir", dir);
            }
            Gravity::Towards { point, strength } => {
                vector(ui, "Point", point);
                scalar(ui, "Strength", strength);
            }
            Gravity::None => {}
            #[cfg(feature = "math")]
            Gravity::VectorField { x, y, funcs } => {
                fn compile_text(ui: &mut egui::Ui, expr: &mut String, ex: &mut ExprRes) -> bool {
                    let res = ui.text_edit_singleline(expr);
                    let mut changed = false;
                    if res.changed() {
                        match Expr::try_from(expr.as_str()).map(|e| e.simplify()) {
                            Ok(e) => {
                                *ex = ExprRes::Expr(e);
                                changed = true;
                            }
                            Err(e) => {
                                *ex = ExprRes::Error(e);
                            }
                        }
                    }
                    if let ExprRes::Error(e) = ex {
                        for e in e {
                            ui.colored_label(Rgba::RED, format!("{:?}", e));
                        }
                    }
                    changed
                }
                let mut changed = compile_text(ui, x, &mut state.expr_x);
                if let Some(err) = &state.x_compile_err {
                    ui.colored_label(Rgba::RED, format!("{}", err));
                }
                changed |= compile_text(ui, y, &mut state.expr_y);
                if let Some(err) = &state.y_compile_err {
                    ui.colored_label(Rgba::RED, format!("{}", err));
                }
                if changed && let (ExprRes::Expr(x), ExprRes::Expr(y)) = (&state.expr_x, &state.expr_y) {
                    let args = &[Identifier::from('x'), Identifier::from('y')];
                    match (compile(x, args), compile(y, args)) {
                        (Ok(x), Ok(y)) => {
                            *funcs = Some((x, y));
                            state.x_compile_err = None;
                            state.y_compile_err = None;
                        }
                        (Err(e), Ok(_)) => {
                            state.x_compile_err = Some(e);
                            state.y_compile_err = None;
                        }
                        (Ok(_), Err(e)) => {
                            state.x_compile_err = None;
                            state.y_compile_err = Some(e);
                        }
                        (Err(e0), Err(e1)) => {
                            state.x_compile_err = Some(e0);
                            state.y_compile_err = Some(e1);
                        }
                    }
                }
            }
        }

        ui.label("Gravitational Constant");
        ui.add(egui::DragValue::new(&mut settings.gravitational_constant));

        ui.checkbox(&mut settings.collisions, "Collisions");

        if ui.button("Stop All").clicked() {
            objects.iter_mut().for_each(|(_, mut o)| o.old = o.current);
        }
        if ui.button("Prune").clicked() {
            objects.iter().for_each(|(e, pos)| if pos.current.length_squared() > 1_000_000_000.0 {
                commands.entity(e).despawn();
            })
        }
        if ui.button("Clear").clicked() {
            objects.iter().for_each(|(e, _)| commands.entity(e).despawn())
        }
        if ui.button("Remove Points").clicked() {
            points.iter().for_each(|(e, _)| commands.entity(e).despawn())
        }
        ui.horizontal(|ui| {
            ui.label("Sub Steps");
            let mut value = u32::from(settings.sub_steps);
            ui.add(egui::Slider::new(&mut value, 1..=32));
            if let Some(value) = NonZeroU32::new(value) {
                settings.sub_steps = value;
            }
        });

        ui.label(format!("Bodies: {}", objects.iter().count()));
        ui.label(format!("Links: {}", links.iter().count()));

        let fps_diags = diagnostics
            .get(FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|fps| fps.average());
        if let Some(fps) = fps_diags {
            ui.label(format!("FPS: {:.0}", fps));
        }
    });
}
