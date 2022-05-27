use std::ops::IndexMut;

use bevy::{math::Vec2, prelude::*};

use super::PhysObject;

pub trait Constraint<E> {
    type This<U>;
    fn apply<'a, C: IndexMut<E, Output = PhysObject>>(&self, get: &mut C);
    fn should_stay<F: Fn(&E) -> Vec2>(&self, get: F) -> bool;
    fn try_map<T, F: FnMut(&E) -> Option<T>>(&self, map: F) -> Option<Self::This<T>>;
}

#[derive(Component)]
pub struct LinkConstraint<E> {
    a: E,
    b: E,
    dist: f32,
    snap: f32,
}

impl LinkConstraint<Entity> {
    pub fn new(a: Entity, b: Entity, dist: f32, snap: f32) -> Self {
        Self { a, b, dist, snap }
    }
}

impl<E: Copy> Constraint<E> for LinkConstraint<E> {
    type This<U> = LinkConstraint<U>;

    fn apply<'a, C: IndexMut<E, Output = PhysObject>>(&self, get: &mut C) {
        let axis = get[self.a].pos - get[self.b].pos;
        let axis = if axis == Vec2::ZERO {
            Vec2::new(f32::EPSILON, f32::EPSILON)
        } else {
            axis
        };
        let dist = axis.length();
        let n = axis / dist;
        let delta = self.dist - dist;
        let a_m = get[self.a].mass;
        let b_m = get[self.b].mass;
        get[self.a].pos += n * (delta * b_m / (a_m + b_m));
        get[self.b].pos -= n * (delta * a_m / (a_m + b_m));
        #[cfg(feature = "panic-nan")]
        {
            get[self.a].panic_nan("link");
            get[self.b].panic_nan("link");
        }
    }

    fn try_map<T, F: FnMut(&E) -> Option<T>>(&self, mut map: F) -> Option<Self::This<T>> {
        map(&self.a).and_then(|a| {
            map(&self.b).map(|b| LinkConstraint {
                a,
                b,
                dist: self.dist,
                snap: self.snap,
            })
        })
    }

    fn should_stay<F: Fn(&E) -> Vec2>(&self, get: F) -> bool {
        get(&self.a).distance_squared(get(&self.b)) < self.snap * self.snap
    }
}

#[derive(Component)]
pub struct PointConstraint<E> {
    a: E,
    dist: f32,
    point: Vec2,
}

impl<E> PointConstraint<E> {
    pub fn try_map<T>(&self, mut map: impl FnMut(&E) -> Option<T>) -> Option<PointConstraint<T>> {
        map(&self.a).map(|a| PointConstraint {
            a,
            dist: self.dist,
            point: self.point,
        })
    }
}

impl PointConstraint<Entity> {
    pub fn new(a: Entity, point: Vec2, dist: f32) -> Self {
        Self { a, point, dist }
    }
}

impl<E: Copy> Constraint<E> for PointConstraint<E> {
    type This<U> = PointConstraint<U>;

    fn apply<'a, C: IndexMut<E, Output = PhysObject>>(&self, get: &mut C) {
        let axis = get[self.a].pos - self.point;
        let dist = axis.length();
        let n = axis / dist;
        let delta = self.dist - dist;
        let p = self.point + n * delta;
        get[self.a].pos = p;
        get[self.a].pos_old = p;
        get[self.a].acceleration = Vec2::ZERO;
        #[cfg(feature = "panic-nan")]
        get[self.a].panic_nan("point");
    }

    fn try_map<T, F: FnMut(&E) -> Option<T>>(&self, mut map: F) -> Option<Self::This<T>> {
        map(&self.a).map(|a| PointConstraint {
            a,
            dist: self.dist,
            point: self.point,
        })
    }

    fn should_stay<F: Fn(&E) -> Vec2>(&self, _: F) -> bool {
        true
    }
}
