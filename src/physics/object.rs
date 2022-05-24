use bevy::{math::DVec2, prelude::*};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum Material {
    Metal,
    Wood,
    Water,
    BlackHole,
}

impl Material {
    fn color(&self) -> Color {
        match self {
            Material::Metal => Color::WHITE,
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
    pub material: Material,
    pub radius: f64,
}

#[derive(Bundle)]
pub struct ObjectBundle {
    object: Object,
    pos: ObjectPos,
    sprite: Sprite,
    transform: Transform,
    global_transform: GlobalTransform,
    texture: Handle<Image>,
    /// User indication of whether an entity is visible
    visibility: Visibility,
}

impl ObjectBundle {
    pub fn new(pos: DVec2, material: Material, radius: f64, image: Handle<Image>) -> Self {
        Self {
            object: Object { material, radius },
            pos: ObjectPos {
                current: pos,
                old: pos,
            },
            sprite: Sprite {
                color: material.color(),
                custom_size: Some(Vec2::ONE),
                ..default()
            },
            transform: Transform {
                translation: Vec3::new(pos.x as f32, pos.y as f32, 0.0),
                scale: Vec3::splat(radius as f32 * 2.0),
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

impl From<(&Object, &ObjectPos)> for PhysObject {
    fn from((obj, pos): (&Object, &ObjectPos)) -> Self {
        PhysObject {
            pos: pos.current,
            pos_old: pos.old,
            acceleration: DVec2::ZERO,
            radius: obj.radius,
            mass: obj.radius * obj.radius * obj.material.density() * std::f64::consts::PI,
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
    pub fn accelerate(&mut self, acc: DVec2) {
        self.acceleration += acc;
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
        sprite.color = obj.material.color();
        transform.scale = Vec3::splat(obj.radius as f32 * 2.0);
    });
}
