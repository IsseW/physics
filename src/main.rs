#![feature(let_chains)]
use std::num::NonZeroU32;

use bevy::{
    diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin},
    math::{DVec2, Vec3Swizzles, Vec4Swizzles},
    prelude::*,
};
use bevy_egui::{
    egui::{self, panel::Side, DragValue, Rgba},
    EguiContext, EguiPlugin,
};
use bevy_pancam::{PanCam, PanCamPlugin};
use massi::{
    cranelift::{compile, CFunc, ModuleError},
    parser::{Error, Full},
    Identifier,
};
use rand::Rng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[cfg(feature = "math")]
use massi::Expr;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugin(FrameTimeDiagnosticsPlugin)
        .add_plugin(PanCamPlugin)
        .init_resource::<PhysWorld>()
        .insert_resource(ClearColor(Color::BLACK))
        .add_startup_system(load_system)
        .add_system(physics_system)
        .add_system(input_system)
        .add_system(extract_system)
        .add_system(ui)
        .run();
}

enum ExprRes {
    Expr(Expr),
    Error(Vec<Error<Full>>),
}

impl Default for ExprRes {
    fn default() -> Self {
        Self::Expr(Expr::empty())
    }
}

#[derive(Default)]
struct State {
    expr_x: ExprRes,
    x_compile_err: Option<ModuleError>,
    expr_y: ExprRes,
    y_compile_err: Option<ModuleError>,
}

fn ui(
    mut egui_context: ResMut<EguiContext>,
    mut world: ResMut<PhysWorld>,
    diagnostics: Res<Diagnostics>,
    mut state: Local<State>,
) {
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

        let mut curr = world.bounds.as_str();
        egui::ComboBox::from_label("Bounds")
            .selected_text(curr)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut curr, "Circle", "Circle");
                ui.selectable_value(&mut curr, "Rect", "Rect");
                ui.selectable_value(&mut curr, "None", "None");
            });

        world.bounds = Bounds::from_str(curr, world.bounds.clone());

        match &mut world.bounds {
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

        let mut curr = world.gravity.as_str();
        egui::ComboBox::from_label("Gravity")
            .selected_text(curr)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut curr, "Dir", "Dir");
                ui.selectable_value(&mut curr, "Towards", "Towards");
                #[cfg(feature = "math")]
                ui.selectable_value(&mut curr, "Vector Field", "Vector Field");
                ui.selectable_value(&mut curr, "None", "None");
            });
        world.gravity = Gravity::from_str(curr, world.gravity.clone());

        match &mut world.gravity {
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
                    if !res.has_focus() && let ExprRes::Expr(ex) = ex {
                        *expr = format!("{ex}");
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
        ui.add(egui::DragValue::new(&mut world.gravitational_constant));

        if ui.button("Stop All").clicked() {
            world.objects.iter_mut().for_each(|o| o.pos_old = o.pos);
        }
        ui.horizontal(|ui| {
            ui.label("Sub Steps");
            let mut value = u32::from(world.sub_steps);
            ui.add(egui::Slider::new(&mut value, 1..=32));
            if let Some(value) = NonZeroU32::new(value) {
                world.sub_steps = value;
            }
        });

        ui.label(format!("Bodies: {}", world.objects.len()));

        let fps_diags = diagnostics
            .get(FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|fps| fps.average());
        if let Some(fps) = fps_diags {
            ui.label(format!("FPS: {:.0}", fps));
        }
    });
}

fn load_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(Circle(asset_server.load("circle.png")));

    commands
        .spawn_bundle(OrthographicCameraBundle::new_2d())
        .insert(PanCam {
            grab_buttons: vec![MouseButton::Middle],
            enabled: true,
        });
}

#[derive(Clone)]
enum Material {
    Metal,
    Wood,
    Water,
    BlackHole,
}

impl Material {
    fn color(&self) -> Color {
        match self {
            Material::Metal => Color::GRAY,
            Material::Wood => Color::BEIGE,
            Material::Water => Color::BLUE,
            Material::BlackHole => Color::rgb(0.18, 0.1, 0.2),
        }
    }

    fn density(&self) -> f64 {
        match self {
            Material::Metal => 7.874,
            Material::Wood => 0.1,
            Material::Water => 1.0,
            Material::BlackHole => 1000.0,
        }
    }
}

#[derive(Clone)]
struct Object {
    pos: DVec2,
    pos_old: DVec2,
    acceleration: DVec2,

    radius: f64,

    // This is dependendant on material and radius
    mass: f64,

    // Should not change during the lifetime of an object.
    material: Material,
    binding: Option<Entity>,
}

impl Object {
    fn new(pos: DVec2, radius: f64, material: Material) -> Self {
        let mass = radius * radius * material.density() * std::f64::consts::PI;
        Self {
            pos,
            pos_old: pos,
            acceleration: DVec2::ZERO,
            radius,
            mass,
            material,
            binding: None,
        }
    }

    #[inline(always)]
    fn update_position(&mut self, dt: f64) {
        let velocity = if !self.pos.is_finite() {
            if self.pos_old.is_finite() {
                self.pos = self.pos_old;
            } else {
                self.pos = DVec2::ZERO;
            }
            DVec2::ZERO
        } else {
            self.pos - self.pos_old
        };
        self.pos_old = self.pos;
        self.pos += velocity + self.acceleration * dt * dt;
        self.acceleration = DVec2::ZERO;
    }

    #[inline(always)]
    fn accelerate(&mut self, acc: DVec2) {
        self.acceleration += acc;
    }
}

#[derive(Clone)]
enum Gravity {
    Dir(DVec2),
    Towards {
        point: DVec2,
        strength: f64,
    },
    #[cfg(feature = "math")]
    VectorField {
        funcs: Option<(CFunc<2>, CFunc<2>)>,
        x: String,
        y: String,
    },
    None,
}

impl Gravity {
    #[inline(always)]
    fn acceleration(&self, pos: DVec2) -> DVec2 {
        fn towards(point: DVec2, strength: f64, pos: DVec2) -> DVec2 {
            let axis = point - pos;
            let sqr_len = axis.length_squared();
            let len = sqr_len.sqrt();
            axis * (strength / (sqr_len * len))
        }
        match self {
            Gravity::Dir(dir) => *dir,
            Gravity::Towards { point, strength } => towards(*point, *strength, pos),
            // Per object is applied at a later stage
            Gravity::None => DVec2::ZERO,
            #[cfg(feature = "math")]
            Gravity::VectorField { funcs, .. } => {
                if let Some(funcs) = funcs {
                    let pos = &[pos.x, pos.y];
                    DVec2::new(funcs.0(pos), funcs.1(pos))
                } else {
                    DVec2::ZERO
                }
            }
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Gravity::Dir(_) => "Dir",
            Gravity::Towards { .. } => "Towards",
            Gravity::None => "None",
            #[cfg(feature = "math")]
            Gravity::VectorField { .. } => "Vector Field",
        }
    }

    fn from_str(s: &str, gravity: Gravity) -> Gravity {
        match s {
            "Dir" => {
                if matches!(gravity, Gravity::Dir(_)) {
                    gravity
                } else {
                    Gravity::Dir(DVec2::new(0.0, -400.0))
                }
            }
            "Towards" => {
                if matches!(gravity, Gravity::Towards { .. }) {
                    gravity
                } else {
                    Gravity::Towards {
                        point: DVec2::new(0.0, 0.0),
                        strength: 100.0,
                    }
                }
            }
            "None" => Gravity::None,
            #[cfg(feature = "math")]
            "Vector Field" => {
                if matches!(gravity, Gravity::VectorField { .. }) {
                    gravity
                } else {
                    Gravity::VectorField {
                        funcs: None,
                        x: String::new(),
                        y: String::new(),
                    }
                }
            }
            _ => gravity,
        }
    }
}

#[derive(Clone)]
enum Bounds {
    Circle(f64),
    Rect(DVec2, DVec2),
    None,
}

impl Bounds {
    #[inline(always)]
    fn random_point(&self, rng: &mut impl Rng, radius: f64) -> DVec2 {
        match self {
            Bounds::Circle(r) => {
                let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                let r = rng.gen_range(0.0..r - radius);
                DVec2::new(angle.cos() * r, angle.sin() * r)
            }
            Bounds::Rect(min, max) => {
                let x = rng.gen_range(min.x + radius..max.x - radius);
                let y = rng.gen_range(min.y + radius..max.y - radius);
                DVec2::new(x, y)
            }
            Bounds::None => DVec2::new(0.0, 0.0),
        }
    }

    #[inline(always)]
    fn update_position(&self, obj: &mut Object) {
        match self {
            Bounds::Circle(r) => {
                let l = obj.pos.length();
                if l > r - obj.radius {
                    obj.pos = obj.pos / l * (r - obj.radius);
                }
            }
            Bounds::Rect(min, max) => {
                let r = DVec2::splat(obj.radius);
                obj.pos = obj.pos.clamp(*min + r, *max - r);
            }
            Bounds::None => {}
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Bounds::Circle(_) => "Circle",
            Bounds::Rect(_, _) => "Rect",
            Bounds::None => "None",
        }
    }

    fn from_str(s: &str, bounds: Bounds) -> Bounds {
        match s {
            "Circle" => {
                if matches!(bounds, Bounds::Circle(_)) {
                    bounds
                } else {
                    Bounds::Circle(200.0)
                }
            }
            "Rect" => {
                if matches!(bounds, Bounds::Rect(_, _)) {
                    bounds
                } else {
                    Bounds::Rect(DVec2::new(-100.0, -100.0), DVec2::new(100.0, 100.0))
                }
            }
            "None" => Bounds::None,
            _ => bounds,
        }
    }
}

struct PhysWorld {
    objects: Vec<Object>,
    gravity: Gravity,
    bounds: Bounds,
    gravitational_constant: f64,
    sub_steps: NonZeroU32,
}

impl Default for PhysWorld {
    fn default() -> Self {
        Self {
            objects: Default::default(),
            gravity: Gravity::None,
            bounds: Bounds::None,
            gravitational_constant: Default::default(),
            sub_steps: NonZeroU32::new(1).unwrap(),
        }
    }
}

impl PhysWorld {
    fn for_pairs<T: Send>(
        &mut self,
        map: impl Fn(&Object, &Object) -> Option<(T, T)> + Sync,
        mut func: impl FnMut(&mut Object, T),
    ) {
        self.objects
            .par_iter()
            .enumerate()
            .map(|(i, o0)| {
                self.objects[i + 1..]
                    .iter()
                    .enumerate()
                    .filter_map(|(j, o1)| map(o0, o1).map(|(a, b)| ((i, a), (i + 1 + j, b))))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|v| v.into_iter())
            .for_each(|((i, a), (j, b))| {
                func(&mut self.objects[i], a);
                func(&mut self.objects[j], b);
            });
    }

    #[inline(always)]
    fn tick(&mut self, dt: f64) {
        #[cfg(feature = "tracy")]
        profiling::scope!("physics system");
        let sub_steps = u32::from(self.sub_steps);
        let dt = dt / sub_steps as f64;
        for _ in 0..sub_steps {
            #[cfg(feature = "tracy")]
            profiling::scope!("tick");
            // Handle gravity
            {
                #[cfg(feature = "tracy")]
                profiling::scope!("gravity");

                if self.gravitational_constant.abs() > f64::EPSILON {
                    let constant = self.gravitational_constant;
                    // self.for_pairs(
                    //     |a, b| {
                    //         let axis = a.pos - b.pos;
                    //         let sqr_len = axis.length_squared();
                    //         let n = axis * (constant / sqr_len / sqr_len.sqrt());
                    //         Some((n * b.mass, n * -a.mass))
                    //     },
                    //     |o, t| {
                    //         o.accelerate(t);
                    //     },
                    // );
                    let objects = self.objects.clone();
                    self.objects.par_iter_mut().for_each(|a| {
                        let v = objects
                            .iter()
                            .map(|b| {
                                let axis = b.pos - a.pos;
                                let sqr_len = axis.length_squared();
                                if sqr_len == 0.0 {
                                    DVec2::ZERO
                                } else {
                                    axis * (b.mass * constant / sqr_len / sqr_len.sqrt())
                                }
                            })
                            .reduce(|a, b| a + b)
                            .unwrap_or_default();
                        a.accelerate(v);
                    });
                }
                self.objects
                    .iter_mut()
                    .for_each(|obj| obj.accelerate(self.gravity.acceleration(obj.pos)));
            }

            // Handle bounds
            {
                #[cfg(feature = "tracy")]
                profiling::scope!("bounds");
                self.objects.iter_mut().for_each(|obj| {
                    self.bounds.update_position(obj);
                });
            }

            // Handle collisions
            {
                #[cfg(feature = "tracy")]
                profiling::scope!("collisions");
                self.for_pairs(
                    |a, b| {
                        let collision_axis = a.pos - b.pos;
                        let combined = a.radius + b.radius;
                        let dist_sqr = collision_axis.length_squared();
                        if dist_sqr < combined * combined {
                            let dist = dist_sqr.sqrt();
                            let n = collision_axis / dist;
                            let delta = combined - dist;
                            Some((
                                (b.mass / (a.mass + b.mass) * delta) * n,
                                -(a.mass / (a.mass + b.mass) * delta) * n,
                            ))
                        } else {
                            None
                        }
                    },
                    |o, t| {
                        o.pos += t;
                    },
                );
            }

            // Update positions
            {
                #[cfg(feature = "tracy")]
                profiling::scope!("update");
                self.objects.iter_mut().for_each(|obj| {
                    obj.update_position(dt);
                });
            }
        }
    }
}

struct Circle(Handle<Image>);

fn input_system(
    mut phys_world: ResMut<PhysWorld>,
    input: Res<Input<KeyCode>>,
    windows: Res<Windows>,
    camera: Query<(&Camera, &GlobalTransform)>,
) {
    if let Some(window) = windows.get_primary() && let Some(position) = window.cursor_position() {
        // cursor is inside the window, position given
        if let Ok((camera, transform)) = camera.get_single() {
            let position = (position / Vec2::new(window.width(), window.height()) - 0.5) * 2.0;
            let pos = transform.mul_vec3((camera.projection_matrix.inverse()
                * Vec4::new(position.x, position.y, 0.0, 1.0)).xyz())
            .xy()
            .as_dvec2();
            let mut rng = rand::thread_rng();
            if input.pressed(KeyCode::Space) {
                phys_world
                    .objects
                    .push(Object::new(pos, rng.gen_range(4.0..10.0), Material::Metal))
            }
            if input.just_pressed(KeyCode::B) {
                phys_world
                    .objects
                    .push(Object::new(pos, rng.gen_range(400.0..1000.0), Material::BlackHole))
            }
            if input.pressed(KeyCode::G) {
                phys_world.gravity = Gravity::Towards {
                    point: pos,
                    strength: if let Gravity::Towards { point: _, strength } = phys_world.gravity {
                        strength
                    } else {
                        400.0
                    },
                };
            }
        }
    }
}

fn physics_system(mut phys_world: ResMut<PhysWorld>, time: Res<Time>) {
    phys_world.tick(time.delta_seconds_f64());
}

fn extract_system(
    mut commands: Commands,
    mut world: ResMut<PhysWorld>,
    mut objects: Query<&mut Transform>,
    circle: Res<Circle>,
) {
    #[cfg(feature = "tracy")]
    profiling::scope!("extract system");
    world.objects.iter_mut().for_each(|o| {
        if let Some(Ok(mut transform)) = o.binding.map(|e| objects.get_mut(e)) {
            transform.translation =
                Vec3::new(o.pos.x as f32, o.pos.y as f32, transform.translation.z);
            transform.scale = Vec3::splat(o.radius as f32 * 2.0);
        } else {
            let e = commands
                .spawn_bundle(SpriteBundle {
                    sprite: Sprite {
                        color: o.material.color(),
                        custom_size: Some(Vec2::ONE),
                        ..default()
                    },
                    transform: Transform {
                        translation: Vec3::new(o.pos.x as f32, o.pos.y as f32, 0.0),
                        scale: Vec3::splat(o.radius as f32 * 2.0),
                        ..default()
                    },
                    texture: circle.0.clone(),
                    ..default()
                })
                .id();
            o.binding = Some(e);
        }
    })
}
