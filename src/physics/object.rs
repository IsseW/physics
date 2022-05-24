use bevy::{math::DVec2, prelude::*};

use crate::PlacementSettings;

#[derive(Component)]
pub struct ObjectDensity(f64);

#[derive(Component)]
pub struct ObjectPos {
    pub current: DVec2,
    pub old: DVec2,
}

impl ObjectPos {
    pub fn apply(&mut self, obj: PhysObject) {
        self.current = obj.pos;
        self.old = obj.pos_old;
    }
}

#[derive(Component)]
pub struct Object {
    pub color: Color,
    pub radius: f64,
}

#[derive(Bundle)]
pub struct ObjectBundle {
    object: Object,
    pos: ObjectPos,
    density: ObjectDensity,
    sprite: Sprite,
    transform: Transform,
    global_transform: GlobalTransform,
    texture: Handle<Image>,
    /// User indication of whether an entity is visible
    visibility: Visibility,
}

impl ObjectBundle {
    pub fn new(pos: DVec2, settings: &PlacementSettings, image: Handle<Image>) -> Self {
        Self {
            object: Object { color: settings.color, radius: settings.radius },
            pos: ObjectPos {
                current: pos,
                old: pos,
            },
            density: ObjectDensity(settings.density),
            sprite: Sprite {
                color: settings.color,
                custom_size: Some(Vec2::ONE),
                ..default()
            },
            transform: Transform {
                translation: Vec3::new(pos.x as f32, pos.y as f32, 0.0),
                scale: Vec3::splat(settings.radius as f32 * 2.0),
                ..default()
            },
            texture: image,
            global_transform: Default::default(),
            visibility: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct PhysObject {
    pub(super) pos: DVec2,
    pub(super) pos_old: DVec2,
    pub(super) acceleration: DVec2,
    pub(super) radius: f64,
    pub(super) mass: f64,
}

impl From<(&Object, &ObjectPos, &ObjectDensity)> for PhysObject {
    fn from((obj, pos, density): (&Object, &ObjectPos, &ObjectDensity)) -> Self {
        PhysObject {
            pos: pos.current,
            pos_old: pos.old,
            acceleration: DVec2::ZERO,
            radius: obj.radius,
            mass: obj.radius * obj.radius * density.0 * std::f64::consts::PI,
        }
    }
}

impl PhysObject {
    #[inline(always)]
    pub fn has_changed(&self) -> bool {
        self.pos != self.pos_old
    }

    #[inline(always)]
    pub fn update_position(&mut self, dt: f64) {
        #[cfg(feature = "panic-nan")]
        self.panic_nan("pre update");
        let velocity = self.pos - self.pos_old;
        self.pos_old = self.pos;
        self.pos += velocity + self.acceleration * dt * dt;
        self.acceleration = DVec2::ZERO;
        #[cfg(feature = "panic-nan")]
        self.panic_nan("post update");
    }

    
    #[inline(always)]
    pub fn set_velocity(&mut self, vel: DVec2) {
        self.pos_old = self.pos - vel;
    }

    #[inline(always)]
    pub fn accelerate(&mut self, acc: DVec2) {
        self.acceleration += acc;
    }

    #[cfg(feature = "panic-nan")]
    pub fn panic_nan(&self, msg: &str) {
        if self.acceleration.is_nan() {
            panic!("{} NAN acceleration", msg);
        }
        if self.pos.is_nan() {
            panic!("{} NAN position", msg);
        }
        if self.pos_old.is_nan() {
            panic!("{} NAN position_old", msg);
        }
    }
}

pub(super) fn update_position_system(
    mut positions: Query<(&ObjectPos, &mut Transform), Changed<ObjectPos>>,
) {
    #[cfg(feature = "tracy")]
    profiling::scope!("update position system");
    positions.for_each_mut(|(pos, mut transform)| {
        transform.translation = Vec3::new(
            pos.current.x as f32,
            pos.current.y as f32,
            transform.translation.z,
        );
    });
}

pub(super) fn update_visuals_system(
    mut objects: Query<(&Object, &mut Transform, &mut Sprite), Changed<Object>>,
) {
    #[cfg(feature = "tracy")]
    profiling::scope!("update visuals system");
    objects.for_each_mut(|(obj, mut transform, mut sprite)| {
        sprite.color = obj.color;
        transform.scale = Vec3::splat(obj.radius as f32 * 2.0);
    });
}
