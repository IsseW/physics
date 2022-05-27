mod constraints;
mod object;

use std::{f32::consts::PI, num::NonZeroU32};

use self::object::ObjectDensity;
pub use self::{
    constraints::{LinkConstraint, PointConstraint},
    object::{Object, ObjectBundle, ObjectPos, PhysObject},
};

use crate::{for_pairs::ForPairs, physics::constraints::Constraint};
use bevy::{math::Vec2, prelude::*, utils::HashMap};
#[cfg(feature = "math")]
use massi::cranelift::CFunc;
use rand::Rng;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

#[derive(Clone)]
pub enum Gravity {
    Dir(Vec2),
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
    fn acceleration(&self, pos: Vec2) -> Vec2 {
        match self {
            Gravity::Dir(dir) => *dir,
            // Per object is applied at a later stage
            Gravity::None => Vec2::ZERO,
            #[cfg(feature = "math")]
            Gravity::VectorField { funcs, .. } => {
                if let Some(funcs) = funcs {
                    let pos = &[pos.x as f64, pos.y as f64];
                    Vec2::new(funcs.0(pos) as f32, funcs.1(pos) as f32)
                } else {
                    Vec2::ZERO
                }
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Gravity::Dir(_) => "Dir",
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
                    Gravity::Dir(Vec2::new(0.0, -400.0))
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
    Circle(f32),
    Rect(Vec2, Vec2),
    None,
}

impl Bounds {
    #[allow(dead_code)]
    #[inline(always)]
    fn random_point(&self, rng: &mut impl Rng, radius: f32) -> Vec2 {
        match self {
            Bounds::Circle(r) => {
                let angle = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
                let r = rng.gen_range(0.0..r - radius);
                Vec2::new(angle.cos() * r, angle.sin() * r)
            }
            Bounds::Rect(min, max) => {
                let x = rng.gen_range(min.x + radius..max.x - radius);
                let y = rng.gen_range(min.y + radius..max.y - radius);
                Vec2::new(x, y)
            }
            Bounds::None => Vec2::new(0.0, 0.0),
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
                let r = Vec2::splat(obj.radius);
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
                    Bounds::Rect(Vec2::new(-100.0, -100.0), Vec2::new(100.0, 100.0))
                }
            }
            "None" => Bounds::None,
            _ => bounds,
        }
    }
}

pub struct PhysSettings {
    pub gravity: Gravity,
    pub gravity_set_velocity: bool,
    pub bounds: Bounds,
    pub gravitational_constant: f32,
    pub sub_steps: NonZeroU32,
    pub collisions: bool,
    pub use_simd: bool,
}

impl Default for PhysSettings {
    fn default() -> Self {
        Self {
            gravity: Gravity::None,
            gravity_set_velocity: false,
            bounds: Bounds::None,
            gravitational_constant: Default::default(),
            sub_steps: NonZeroU32::new(1).unwrap(),
            collisions: true,
            use_simd: false,
        }
    }
}

fn physics_system(
    mut commands: Commands,
    mut objects: Query<(Entity, (&Object, &mut ObjectPos, &ObjectDensity))>,
    links: Query<(Entity, &LinkConstraint<Entity>)>,
    points: Query<(Entity, &PointConstraint<Entity>)>,
    settings: Res<PhysSettings>,
    time: Res<Time>,
) {
    #[cfg(feature = "tracy")]
    profiling::scope!("physics system");
    if settings.use_simd {
        use core::arch::x86_64::*;

        trait ObjectCollection {
            fn len(&self) -> usize;
            fn pos(&self, i: usize) -> Vec2;
        }

        trait ObjectCollectionMut: ObjectCollection {
            fn set_pos(&mut self, i: usize, pos: Vec2);
            fn set_velocity(&mut self, i: usize, vel: Vec2);
            fn accelerate(&mut self, i: usize, acc: Vec2);
        }

        struct ObjectsChunkMut<'a> {
            x: &'a mut [f32],
            y: &'a mut [f32],
            oldx: &'a mut [f32],
            oldy: &'a mut [f32],
            mass: &'a mut [f32],
            radius: &'a mut [f32],
            accx: &'a mut [f32],
            accy: &'a mut [f32],
        }

        impl<'a> ObjectCollection for ObjectsChunkMut<'a> {
            fn len(&self) -> usize {
                self.x.len()
            }

            fn pos(&self, i: usize) -> Vec2 {
                Vec2::new(self.x[i], self.y[i])
            }
        }

        impl<'a> ObjectCollectionMut for ObjectsChunkMut<'a> {
            fn set_pos(&mut self, i: usize, pos: Vec2) {
                self.x[i] = pos.x;
                self.y[i] = pos.y;
            }

            fn set_velocity(&mut self, i: usize, vel: Vec2) {
                self.oldx[i] = self.x[i] - vel.x;
                self.oldy[i] = self.y[i] - vel.y;
            }

            fn accelerate(&mut self, i: usize, acc: Vec2) {
                self.accx[i] += acc.x;
                self.accy[i] += acc.y;
            }
        }

        struct ObjectsChunk<'a> {
            x: &'a [f32],
            y: &'a [f32],
            oldx: &'a [f32],
            oldy: &'a [f32],
            mass: &'a [f32],
            radius: &'a [f32],
            accx: &'a [f32],
            accy: &'a [f32],
        }
        impl<'a> ObjectCollection for ObjectsChunk<'a> {
            fn len(&self) -> usize {
                self.x.len()
            }

            fn pos(&self, i: usize) -> Vec2 {
                Vec2::new(self.x[i], self.y[i])
            }
        }
        struct PhysObjects {
            count: usize,
            x: Vec<f32>,
            y: Vec<f32>,
            oldx: Vec<f32>,
            oldy: Vec<f32>,
            mass: Vec<f32>,
            radius: Vec<f32>,
            accx: Vec<f32>,
            accy: Vec<f32>,
        }
        impl ObjectCollection for PhysObjects {
            fn len(&self) -> usize {
                self.count
            }

            fn pos(&self, i: usize) -> Vec2 {
                Vec2::new(self.x[i], self.y[i])
            }
        }

        impl PhysObjects {
            fn par_chunks(&self, num: usize) -> impl ParallelIterator<Item = ObjectsChunk> {
                self.x
                    .par_chunks(num)
                    .zip(self.y.par_chunks(num))
                    .zip(self.oldx.par_chunks(num).zip(self.oldy.par_chunks(num)))
                    .zip(self.mass.par_chunks(num).zip(self.radius.par_chunks(num)))
                    .zip(self.accx.par_chunks(num).zip(self.accy.par_chunks(num)))
                    .take((self.count + num - 1) / num)
                    .map(
                        |((((x, y), (oldx, oldy)), (mass, radius)), (accx, accy))| ObjectsChunk {
                            x,
                            y,
                            oldx,
                            oldy,
                            mass,
                            radius,
                            accx,
                            accy,
                        },
                    )
            }

            fn par_chunks_mut(
                &mut self,
                num: usize,
            ) -> impl ParallelIterator<Item = ObjectsChunkMut> {
                self.x
                    .par_chunks_mut(num)
                    .zip(self.y.par_chunks_mut(num))
                    .zip(
                        self.oldx
                            .par_chunks_mut(num)
                            .zip(self.oldy.par_chunks_mut(num)),
                    )
                    .zip(
                        self.mass
                            .par_chunks_mut(num)
                            .zip(self.radius.par_chunks_mut(num)),
                    )
                    .zip(
                        self.accx
                            .par_chunks_mut(num)
                            .zip(self.accy.par_chunks_mut(num)),
                    )
                    .take((self.count + num - 1) / num)
                    .map(|((((x, y), (oldx, oldy)), (mass, radius)), (accx, accy))| {
                        ObjectsChunkMut {
                            x,
                            y,
                            oldx,
                            oldy,
                            mass,
                            radius,
                            accx,
                            accy,
                        }
                    })
            }

            fn chunks(&self, num: usize) -> impl Iterator<Item = ObjectsChunk> {
                self.x
                    .chunks(num)
                    .zip(self.y.chunks(num))
                    .zip(self.oldx.chunks(num).zip(self.oldy.chunks(num)))
                    .zip(self.mass.chunks(num).zip(self.radius.chunks(num)))
                    .zip(self.accx.chunks(num).zip(self.accy.chunks(num)))
                    .take((self.count + num - 1) / num)
                    .map(
                        |((((x, y), (oldx, oldy)), (mass, radius)), (accx, accy))| ObjectsChunk {
                            x,
                            y,
                            oldx,
                            oldy,
                            mass,
                            radius,
                            accx,
                            accy,
                        },
                    )
            }

            fn chunks_mut(&mut self, num: usize) -> impl Iterator<Item = ObjectsChunkMut> {
                self.x
                    .chunks_mut(num)
                    .zip(self.y.chunks_mut(num))
                    .zip(self.oldx.chunks_mut(num).zip(self.oldy.chunks_mut(num)))
                    .zip(self.mass.chunks_mut(num).zip(self.radius.chunks_mut(num)))
                    .zip(self.accx.chunks_mut(num).zip(self.accy.chunks_mut(num)))
                    .take((self.count + num - 1) / num)
                    .map(|((((x, y), (oldx, oldy)), (mass, radius)), (accx, accy))| {
                        ObjectsChunkMut {
                            x,
                            y,
                            oldx,
                            oldy,
                            mass,
                            radius,
                            accx,
                            accy,
                        }
                    })
            }

            fn into_iter(self) -> impl Iterator<Item = PhysObject> {
                self.x
                    .into_iter()
                    .zip(self.y.into_iter())
                    .zip(self.oldx.into_iter().zip(self.oldy.into_iter()))
                    .zip(self.mass.into_iter().zip(self.radius.into_iter()))
                    .zip(self.accx.into_iter().zip(self.accy.into_iter()))
                    .take(self.count)
                    .map(
                        |((((x, y), (oldx, oldy)), (mass, radius)), (accx, accy))| PhysObject {
                            pos: Vec2::new(x, y),
                            pos_old: Vec2::new(oldx, oldy),
                            mass,
                            radius,
                            acceleration: Vec2::new(accx, accy),
                        },
                    )
            }
        
            fn wrap_back(&mut self) {
                fn wrap<T: Copy>(v: &mut Vec<T>, back_len: usize) {
                    let l = v.len();
                    for i in 0..back_len {
                        v[l - back_len + i] = v[i];
                    }
                }
                let c = self.x.len() - self.count;
                wrap(&mut self.x, c);
                wrap(&mut self.y, c);
                wrap(&mut self.oldx, c);
                wrap(&mut self.oldy, c);
                wrap(&mut self.mass, c);
                wrap(&mut self.radius, c);
                wrap(&mut self.accx, c);
                wrap(&mut self.accy, c);
            }
        }

        const NUM_ELEMS: usize = 4;
        let (mut objs, obj_count, entities) = {
            #[cfg(feature = "tracy")]
            profiling::scope!("extract");
            let count = objects.iter().count();
            let len = count + NUM_ELEMS - 1;
            let mut x = Vec::with_capacity(len);
            let mut y = Vec::with_capacity(len);
            let mut oldx = Vec::with_capacity(len);
            let mut oldy = Vec::with_capacity(len);
            let mut mass = Vec::with_capacity(len);
            let mut radius = Vec::with_capacity(len);
            let mut accx = Vec::with_capacity(len);
            let mut accy = Vec::with_capacity(len);
            let mut entities = HashMap::new();

            for (i, (entity, (obj, pos, density))) in objects.iter().enumerate() {
                x.push(pos.current.x);
                y.push(pos.current.y);
                oldx.push(pos.old.x);
                oldy.push(pos.old.y);
                mass.push(obj.radius * obj.radius * density.0 * PI);
                radius.push(obj.radius);
                accx.push(0.0);
                accy.push(0.0);
                entities.insert(entity, i);
            }

            for _ in 0..NUM_ELEMS - 1 {
                x.push(0.0);
                y.push(0.0);
                oldx.push(0.0);
                oldy.push(0.0);
                mass.push(0.0);
                radius.push(0.0);
                accx.push(0.0);
                accy.push(0.0);
            }

            let objs = PhysObjects {
                count,
                x,
                y,
                oldx,
                oldy,
                mass,
                radius,
                accx,
                accy,
            };

            (objs, count, entities)
        };
        let links = links
            .iter()
            .filter_map(|(e, l)| {
                if let Some(link) = l.try_map(|e| entities.get(e).cloned()) && link.should_stay(|&i| Vec2::new(objs.x[i], objs.y[i])) {
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
        let dt = time.delta_seconds() / sub_steps as f32;

        unsafe {
            let dt_mm = _mm_set1_ps(dt);
            let dt2_mm = _mm_set1_ps(dt * dt);
            let const_mm = _mm_set1_ps(settings.gravitational_constant);
            for _ in 0..sub_steps {
                {
                    #[cfg(feature = "tracy")]
                    profiling::scope!("gravity");
                    if settings.gravitational_constant.abs() > f32::EPSILON {
                        #[cfg(feature = "tracy")]
                        profiling::scope!("dynamic");
                        let mut objs_clone = Vec::with_capacity(obj_count);
                        objs.wrap_back();
                        for i in 0..objs.len() {
                            objs_clone.push((
                                _mm_loadu_ps((&mut objs.x[i..]).as_mut_ptr()),
                                _mm_loadu_ps((&mut objs.y[i..]).as_mut_ptr()),
                                _mm_loadu_ps((&mut objs.mass[i..]).as_mut_ptr()),
                            ));
                        }


                        objs.par_chunks_mut(NUM_ELEMS).for_each(|chunk| {
                            let x = _mm_loadu_ps(chunk.x.as_ptr());
                            let y = _mm_loadu_ps(chunk.y.as_ptr());
                            let mut accx = _mm_loadu_ps(chunk.accx.as_ptr());
                            let mut accy = _mm_loadu_ps(chunk.accy.as_ptr());

                            for obj in &objs_clone {
                                let axis_x = _mm_sub_ps(obj.0, x);
                                let axis_y = _mm_sub_ps(obj.1, y);
                                let sqr_len = _mm_add_ps(
                                    _mm_mul_ps(axis_x, axis_x),
                                    _mm_mul_ps(axis_y, axis_y),
                                );
                                let mask = _mm_cmp_ps::<30>(sqr_len, _mm_setzero_ps());

                                let len = _mm_sqrt_ps(sqr_len);
                                // len^3
                                let len3 = _mm_mul_ps(len, sqr_len);

                                let c = _mm_and_ps(
                                    _mm_mul_ps(const_mm, _mm_div_ps(obj.2, len3)),
                                    mask,
                                );

                                let res_x = _mm_mul_ps(axis_x, c);
                                let res_y = _mm_mul_ps(axis_y, c);
                                accx = _mm_add_ps(accx, res_x);
                                accy = _mm_add_ps(accy, res_y);
                            }

                            _mm_storeu_ps(chunk.accx.as_mut_ptr(), accx);
                            _mm_storeu_ps(chunk.accy.as_mut_ptr(), accy);
                        })
                    }

                    if !matches!(settings.gravity, Gravity::None) {
                        #[cfg(feature = "tracy")]
                        profiling::scope!("const");
                        objs.par_chunks_mut(NUM_ELEMS).for_each(|mut objs| {
                            for i in 0..objs.len() {
                                let v = settings.gravity.acceleration(objs.pos(i));
                                if settings.gravity_set_velocity {
                                    objs.set_velocity(i, v * dt);
                                } else {
                                    objs.accelerate(i, v);
                                }
                                #[cfg(feature = "panic-nan")]
                                obj.panic_nan("const gravity");
                            }
                        });
                    }
                }

                // Update positions
                {
                    #[cfg(feature = "tracy")]
                    profiling::scope!("update");
                    objs.chunks_mut(NUM_ELEMS).for_each(|objs| {
                        let x = _mm_loadu_ps(objs.x.as_ptr());
                        let y = _mm_loadu_ps(objs.y.as_ptr());
                        let oldx = _mm_loadu_ps(objs.oldx.as_ptr());
                        let oldy = _mm_loadu_ps(objs.oldy.as_ptr());

                        let accx = _mm_loadu_ps(objs.accx.as_ptr());
                        let accy = _mm_loadu_ps(objs.accy.as_ptr());

                        let vx = _mm_sub_ps(x, oldx);
                        let vy = _mm_sub_ps(y, oldy);

                        let ax = _mm_mul_ps(accx, dt2_mm);
                        let ay = _mm_mul_ps(accy, dt2_mm);

                        let nx = _mm_add_ps(x, _mm_add_ps(vx, ax));
                        let ny = _mm_add_ps(y, _mm_add_ps(vy, ay));

                        _mm_storeu_ps(objs.x.as_mut_ptr(), nx);
                        _mm_storeu_ps(objs.y.as_mut_ptr(), ny);
                        _mm_storeu_ps(objs.oldx.as_mut_ptr(), x);
                        _mm_storeu_ps(objs.oldy.as_mut_ptr(), y);

                        objs.accx.iter_mut().for_each(|x| *x = 0.0);
                        objs.accy.iter_mut().for_each(|y| *y = 0.0);
                    });
                }
            }
        }

        {
            #[cfg(feature = "tracy")]
            profiling::scope!("insert");
            objects
                .iter_mut()
                .zip(objs.into_iter())
                .for_each(|((e, (_, mut p, _)), obj)| {
                    if obj.has_changed() {
                        if obj.pos.is_nan() {
                            commands.entity(e).despawn();
                        }
                        p.apply(obj);
                    }
                });
        }
    } else {
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
                if let Some(link) = l.try_map(|e| entities.get(e).cloned()) && link.should_stay(|i| objs[*i].pos) {
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
        let dt = time.delta_seconds() / sub_steps as f32;
        for _ in 0..sub_steps {
            #[cfg(feature = "tracy")]
            profiling::scope!("tick");
            // Handle gravity
            {
                #[cfg(feature = "tracy")]
                profiling::scope!("gravity");

                if settings.gravitational_constant.abs() > f32::EPSILON {
                    #[cfg(feature = "tracy")]
                    profiling::scope!("dynamic");
                    let constant = settings.gravitational_constant;

                    let objects = objs.clone();
                    objs.par_iter_mut().for_each(|a| {
                        #[cfg(feature = "tracy")]
                        profiling::scope!("object");
                        let v = objects
                            .iter()
                            .map(|b| {
                                let axis = b.pos - a.pos;
                                let sqr_len = axis.length_squared();

                                if sqr_len == 0.0 {
                                    Vec2::ZERO
                                } else {
                                    axis * (b.mass * constant / sqr_len / sqr_len.sqrt())
                                }
                            })
                            .reduce(|a, b| a + b)
                            .unwrap_or_default();
                        if settings.gravity_set_velocity {
                            a.set_velocity(v * dt);
                        } else {
                            a.accelerate(v);
                        }
                        #[cfg(feature = "panic-nan")]
                        a.panic_nan("gravity")
                    });
                }
                if !matches!(settings.gravity, Gravity::None) {
                    #[cfg(feature = "tracy")]
                    profiling::scope!("const");
                    objs.iter_mut().for_each(|obj| {
                        let v = settings.gravity.acceleration(obj.pos);
                        if settings.gravity_set_velocity {
                            obj.set_velocity(v * dt);
                        } else {
                            obj.accelerate(v);
                        }
                        #[cfg(feature = "panic-nan")]
                        obj.panic_nan("const gravity");
                    });
                }
            }

            // Handle bounds
            if !matches!(settings.bounds, Bounds::None) {
                #[cfg(feature = "tracy")]
                profiling::scope!("bounds");
                objs.iter_mut().for_each(|obj| {
                    settings.bounds.update_position(obj);
                    #[cfg(feature = "panic-nan")]
                    obj.panic_nan("bounds");
                });
            }

            {
                #[cfg(feature = "tracy")]
                profiling::scope!("constraints");

                for link in links.iter() {
                    link.apply(&mut objs);
                }

                for point in points.iter() {
                    point.apply(&mut objs);
                }
            }

            // Handle collisions
            if settings.collisions {
                #[cfg(feature = "tracy")]
                profiling::scope!("collisions");

                objs.par_for_pairs(
                    |a, b| {
                        let collision_axis = a.pos - b.pos;
                        let collision_axis = if collision_axis == Vec2::ZERO {
                            Vec2::new(f32::EPSILON, f32::EPSILON)
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
                        #[cfg(feature = "panic-nan")]
                        o.panic_nan("collision");
                    },
                );
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
                .for_each(|((e, (_, mut p, _)), obj)| {
                    if obj.has_changed() {
                        if obj.pos.is_nan() {
                            commands.entity(e).despawn();
                        }
                        p.apply(obj);
                    }
                })
        }
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
