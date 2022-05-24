use bevy::{math::DVec2, prelude::*};

#[derive(Component)]
pub struct LinkConstraint<E> {
    pub a: E,
    pub b: E,
    pub dist: f64,
    pub snap: f64,
}

impl<E> LinkConstraint<E> {
    pub fn try_map<T>(&self, mut map: impl FnMut(&E) -> Option<T>) -> Option<LinkConstraint<T>> {
        map(&self.a).and_then(|a| {
            map(&self.b).map(|b| LinkConstraint {
                a,
                b,
                dist: self.dist,
                snap: self.snap,
            })
        })
    }
}

#[derive(Component)]
pub struct PointConstraint<E> {
    pub a: E,
    pub dist: f64,
    pub point: DVec2,
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
