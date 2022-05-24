#![feature(let_chains)]
mod for_pairs;
mod physics;
mod ui;

use bevy::{
    diagnostic::FrameTimeDiagnosticsPlugin,
    math::{DVec2, Vec3Swizzles, Vec4Swizzles},
    prelude::*,
};
use bevy_egui::EguiPlugin;
use bevy_pancam::{PanCam, PanCamPlugin};
use physics::{Material, Object, ObjectPos, PhysSettings, PhysicsPlugin};
use rand::Rng;

use crate::physics::{Gravity, LinkConstraint, ObjectBundle, PointConstraint};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugin(FrameTimeDiagnosticsPlugin)
        .add_plugin(PanCamPlugin)
        .add_plugin(PhysicsPlugin)
        .insert_resource(ClearColor(Color::BLACK))
        .add_startup_system(load_system)
        .add_system(input_system)
        .add_system(ui::ui)
        .run();
}

fn load_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    let image = asset_server.load("../assets/circle.png");
    commands.insert_resource(Circle(image));
    commands
        .spawn_bundle(OrthographicCameraBundle::new_2d())
        .insert(PanCam {
            grab_buttons: vec![MouseButton::Middle],
            enabled: true,
        });
}

struct Circle(Handle<Image>);

struct Chain {
    start: Option<DVec2>,
    points: Vec<(f64, DVec2)>,
}

fn input_system(
    mut commands: Commands,
    mut settings: ResMut<PhysSettings>,
    objects: Query<(Entity, &ObjectPos, &Object)>,
    input: Res<Input<KeyCode>>,
    windows: Res<Windows>,
    camera: Query<(&Camera, &GlobalTransform)>,
    circle: Res<Circle>,
    mut chain_builder: Local<Option<Chain>>,
) {
    #[cfg(feature = "tracy")]
    profiling::scope!("input system");
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
                let radius = rng.gen_range(4.0..10.0);
                if objects.iter().all(|(_, p, o)| {
                    let r = radius + o.radius;
                    p.current.distance_squared(pos) > r * r
                }) {
                    commands.spawn_bundle(ObjectBundle::new(pos, Material::Metal, radius, circle.0.clone()));
                }
            }
            if input.pressed(KeyCode::Q) {
                objects.iter().find(|(_, p, o)| {
                    p.current.distance_squared(pos) < o.radius * o.radius
                }).map(|(e, _, _)| {
                    commands.entity(e).despawn();
                });
            }
            if input.just_pressed(KeyCode::B) {
                commands.spawn_bundle(ObjectBundle::new(pos, Material::Wood, rng.gen_range(400.0..1000.0), circle.0.clone()));
            }

            if input.just_pressed(KeyCode::C) {
                if input.pressed(KeyCode::LControl) {
                    *chain_builder = Some(Chain {
                        start: Some(pos),
                        points: Vec::new(),
                    });
                } else {
                    *chain_builder = Some(Chain {
                        start: None,
                        points: Vec::new(),
                    });
                }
            }

            if let Some(chain) = &mut *chain_builder {
                let last_pos = chain.points.last().map(|(_, p)| *p).or_else(|| chain.start);
                let distance = last_pos.map(|p| (pos - p).length()).unwrap_or(f64::INFINITY);
                let chain_distance = 6.5;
                if distance > chain_distance {
                    if let Some(last) = last_pos {
                        let inbetween = (distance / chain_distance - 1.0) as u32;
                        let distance = distance / (inbetween + 1) as f64;
                        for i in 0..inbetween {
                            let t = i as f64 / (inbetween + 1) as f64;
                            let p = last * (1.0 - t) + pos * t;
                            chain.points.push((distance, p));
                        }
                        chain.points.push((distance, pos));
                    } else {
                        chain.points.push((distance, pos));
                    }
                }
                if input.just_released(KeyCode::C) {
                    let chain_obj = |pos| ObjectBundle::new(pos, Material::Metal, 3.0, circle.0.clone());
                    let mut last = chain.start.map(|p| {
                        let id = commands.spawn_bundle(chain_obj(p)).id();
                        commands.spawn().insert(PointConstraint {
                            a: id,
                            dist: 0.0,
                            point: p,
                        });
                        id
                    });
                    let mut last_l = None;
                    for (dist, p) in &chain.points {
                        let id = commands.spawn_bundle(chain_obj(*p)).id();
                        if let Some(last) = last {
                            commands.spawn().insert(LinkConstraint {
                                a: last,
                                b: id,
                                dist: *dist,
                                snap: *dist * 10.0,
                            });
                        }
                        last = Some(id);
                        last_l = Some((id, *p));
                    }
                    if let Some((last, p)) = last_l && input.pressed(KeyCode::LControl) {
                        commands.spawn().insert(PointConstraint {
                            a: last,
                            dist: 0.0,
                            point: p,
                        });
                    }
                    *chain_builder = None;
                }
            }
            if input.pressed(KeyCode::G) {
                settings.gravity = Gravity::Towards {
                    point: pos,
                    strength: if let Gravity::Towards { point: _, strength } = settings.gravity {
                        strength
                    } else {
                        400.0
                    },
                };
            }
        }
    }
}
