#![feature(let_chains)]
mod for_pairs;
mod physics;
mod ui;

use std::f64::consts::{FRAC_PI_2, TAU};

use bevy::{
    diagnostic::FrameTimeDiagnosticsPlugin,
    math::{DVec2, Vec3Swizzles, Vec4Swizzles},
    prelude::*,
};
use bevy_egui::EguiPlugin;
use bevy_pancam::{PanCam, PanCamPlugin};
use physics::{Object, ObjectPos, PhysSettings, PhysicsPlugin};
use rand::Rng;

use crate::physics::{LinkConstraint, ObjectBundle, PointConstraint};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugin(FrameTimeDiagnosticsPlugin)
        .add_plugin(PanCamPlugin)
        .add_plugin(PhysicsPlugin)
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(PlacementSettings {
            radius: 4.0,
            color: Color::WHITE,
            density: 1.0,
        })
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

pub struct PlacementSettings {
    radius: f64,
    color: Color,
    density: f64,
}

fn input_system(
    mut commands: Commands,
    settings: ResMut<PhysSettings>,
    placement: Res<PlacementSettings>,
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
                let pos = if settings.collisions && input.pressed(KeyCode::LControl) {
                    let mut moved = false;
                    let mut pos = pos;
                    let mut last_angle: Option<f64> = None;
                    for _ in 0..10 {
                        pos = objects.iter().fold(pos, |pos, (_, p, o)| {
                            let r = placement.radius + o.radius;
                            if pos.distance_squared(p.current) < r * r {
                                moved = true;
                                let a = if let Some(last_angle) = last_angle { last_angle + rng.gen_range(-FRAC_PI_2*1.1..FRAC_PI_2*1.1) } else {rng.gen_range(0.0..TAU)};
                                last_angle = Some(a);
                                p.current + DVec2::new(a.cos(), a.sin()) * r
                            } else {
                                pos
                            }
                        });
                        if !moved {
                            break;
                        }
                    }
                    pos
                } else {
                    pos
                };
                if !settings.collisions || objects.iter().all(|(_, p, o)| {
                    let r = placement.radius + o.radius;
                    p.current.distance_squared(pos) > r * r
                }) {
                    commands.spawn_bundle(ObjectBundle::new(pos, &placement, circle.0.clone()));
                }
            }
            if input.pressed(KeyCode::Q) {
                objects.iter().find(|(_, p, o)| {
                    p.current.distance_squared(pos) < o.radius * o.radius
                }).map(|(e, _, _)| {
                    commands.entity(e).despawn();
                });
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
                let chain_distance = placement.radius * 1.95;
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
                    let chain_obj = |pos| ObjectBundle::new(pos, &placement, circle.0.clone());
                    let mut last = chain.start.map(|p| {
                        let id = commands.spawn_bundle(chain_obj(p)).id();
                        commands.spawn().insert(PointConstraint::new(id, p, 0.0));
                        id
                    });
                    let mut last_l = None;
                    for (dist, p) in &chain.points {
                        let id = commands.spawn_bundle(chain_obj(*p)).id();
                        if let Some(last) = last {
                            commands.spawn().insert(LinkConstraint::new(last, id, *dist * 1.01, *dist * 10.0));
                        }
                        last = Some(id);
                        last_l = Some((id, *p));
                    }
                    if let Some((last, p)) = last_l && input.pressed(KeyCode::LControl) {
                        commands.spawn().insert(PointConstraint::new(last, p, 0.0));
                    }
                    *chain_builder = None;
                }
            }
        }
    }
}
