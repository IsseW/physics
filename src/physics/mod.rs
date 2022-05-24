mod constraints;
mod object;

use std::num::NonZeroU32;

pub use self::{
    constraints::{LinkConstraint, PointConstraint},
    object::{Material, Object, ObjectBundle, ObjectPos, PhysObject},
};

use crate::for_pairs::ForPairs;
use bevy::{math::DVec2, prelude::*, utils::HashMap};
use massi::cranelift::CFunc;
use rand::Rng;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

#[derive(Clone)]
pub enum Gravity {
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

    pub fn as_str(&self) -> &'static str {
        match self {
            Gravity::Dir(_) => "Dir",
            Gravity::Towards { .. } => "Towards",
            Gravity::None => "None",
            #[cfg(feature = "math")]
            Gravity::VectorField { .. } => "Vector Field",
        }
    }

    pub fn from_str(s: &str, gravity: Gravity) -> Gravity {
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
pub enum Bounds {
    Circle(f64),
    Rect(DVec2, DVec2),
    None,
}

impl Bounds {
    #[allow(dead_code)]
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
    fn update_position(&self, obj: &mut PhysObject) {
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

    pub fn as_str(&self) -> &'static str {
        match self {
            Bounds::Circle(_) => "Circle",
            Bounds::Rect(_, _) => "Rect",
            Bounds::None => "None",
        }
    }

    pub fn from_str(s: &str, bounds: Bounds) -> Bounds {
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

pub struct PhysSettings {
    pub gravity: Gravity,
    pub bounds: Bounds,
    pub gravitational_constant: f64,
    pub sub_steps: NonZeroU32,
    pub collisions: bool,
}

impl Default for PhysSettings {
    fn default() -> Self {
        Self {
            gravity: Gravity::None,
            bounds: Bounds::None,
            gravitational_constant: Default::default(),
            sub_steps: NonZeroU32::new(1).unwrap(),
            collisions: true,
        }
    }
}

fn physics_system(
    mut commands: Commands,
    mut objects: Query<(Entity, (&Object, &mut ObjectPos))>,
    links: Query<(Entity, &LinkConstraint<Entity>)>,
    points: Query<(Entity, &PointConstraint<Entity>)>,
    settings: Res<PhysSettings>,
    time: Res<Time>,
) {
    #[cfg(feature = "tracy")]
    profiling::scope!("physics system");

    let (mut objs, entities) = {
        #[cfg(feature = "tracy")]
        profiling::scope!("extract");

        (
            objects
                .iter()
                .map(|(_, o)| PhysObject::from(o))
                .collect::<Vec<_>>(),
            objects
                .iter()
                .enumerate()
                .map(|(i, (e, _))| (e, i))
                .collect::<HashMap<_, _>>(),
        )
    };

    let links = links
        .iter()
        .filter_map(|(e, l)| {
            if let Some(link) = l.try_map(|e| entities.get(e).cloned()) && objs[link.a].pos.distance_squared(objs[link.b].pos) < l.snap * l.snap {
                Some(link)
            } else {
                commands.entity(e).despawn();
                None
            }
        })
        .collect::<Vec<_>>();

    let points = points
        .iter()
        .filter_map(|(e, l)| {
            let res = l.try_map(|e| entities.get(e).cloned());
            if res.is_none() {
                commands.entity(e).despawn();
            }
            res
        })
        .collect::<Vec<_>>();

    let sub_steps = u32::from(settings.sub_steps);
    let dt = time.delta_seconds_f64() / sub_steps as f64;
    for _ in 0..sub_steps {
        #[cfg(feature = "tracy")]
        profiling::scope!("tick");
        // Handle gravity
        {
            #[cfg(feature = "tracy")]
            profiling::scope!("gravity");

            if settings.gravitational_constant.abs() > f64::EPSILON {
                let constant = settings.gravitational_constant;

                let objects = objs.clone();
                objs.par_iter_mut().for_each(|a| {
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
            if !matches!(settings.gravity, Gravity::None) {
                objs.iter_mut()
                    .for_each(|obj| obj.accelerate(settings.gravity.acceleration(obj.pos)));
            }
        }

        // Handle bounds
        if !matches!(settings.bounds, Bounds::None) {
            #[cfg(feature = "tracy")]
            profiling::scope!("bounds");
            objs.iter_mut().for_each(|obj| {
                settings.bounds.update_position(obj);
            });
        }

        // Handle collisions
        if settings.collisions {
            #[cfg(feature = "tracy")]
            profiling::scope!("collisions");

            objs.par_for_pairs(
                |a, b| {
                    let collision_axis = a.pos - b.pos;
                    let collision_axis = if collision_axis == DVec2::ZERO {
                        DVec2::new(f64::EPSILON, f64::EPSILON)
                    } else {
                        collision_axis
                    };
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

        {
            #[cfg(feature = "tracy")]
            profiling::scope!("constraints");

            for link in links.iter() {
                let axis = objs[link.a].pos - objs[link.b].pos;
                let dist = axis.length();
                let n = axis / dist;
                let delta = link.dist - dist;
                let a_m = objs[link.a].mass;
                let b_m = objs[link.b].mass;
                objs[link.a].pos += n * (delta * b_m / (a_m + b_m));
                objs[link.b].pos -= n * (delta * a_m / (a_m + b_m));
            }

            for point in points.iter() {
                let axis = objs[point.a].pos - point.point;
                let dist = axis.length();
                let n = axis / dist;
                let delta = point.dist - dist;
                objs[point.a].pos = point.point + n * delta;
                objs[point.a].pos_old = objs[point.a].pos;
                objs[point.a].acceleration = DVec2::ZERO;
            }
        }

        // Update positions
        {
            #[cfg(feature = "tracy")]
            profiling::scope!("update");
            objs.iter_mut().for_each(|obj| {
                obj.update_position(dt);
            });
        }
    }

    {
        #[cfg(feature = "tracy")]
        profiling::scope!("insert");
        objects
            .iter_mut()
            .zip(objs.into_iter())
            .for_each(|((_, (_, mut p)), obj)| {
                if obj.has_changed() {
                    p.apply(obj);
                }
            })
    }
}

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PhysSettings>()
            .add_system(physics_system)
            .add_system(object::update_position_system)
            .add_system(object::update_visuals_system);
    }
}
