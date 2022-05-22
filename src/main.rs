use bevy::prelude::*;
use bevy_pancam::{PanCam, PanCamPlugin};
use rand::Rng;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

#[derive(Clone)]
enum Material {
    Metal,
    Wood,
    Water,
}

impl Material {
    fn color(&self) -> Color {
        match self {
            Material::Metal => Color::GRAY,
            Material::Wood => Color::BEIGE,
            Material::Water => Color::BLUE,
        }
    }

    fn density(&self) -> f32 {
        match self {
            Material::Metal => 7.874,
            Material::Wood => 0.1,
            Material::Water => 1.0,
        }
    }
}

#[derive(Clone)]
struct Object {
    pos: Vec2,
    pos_old: Vec2,
    acceleration: Vec2,

    radius: f32,

    // Should not change during the lifetime of an object.
    material: Material,
    binding: Option<Entity>,
}

impl Object {
    fn mass(&self) -> f32 {
        self.radius * self.radius * std::f32::consts::PI * self.material.density()
    }

    fn update_position(&mut self, dt: f32) {
        let velocity = self.pos - self.pos_old;
        self.pos_old = self.pos;
        self.pos = self.pos + velocity + self.acceleration * dt * dt;
        self.acceleration = Vec2::ZERO;
    }

    fn accelerate(&mut self, acc: Vec2) {
        self.acceleration += acc;
    }
}

enum Gravity {
    Dir(Vec2),
    Towards { point: Vec2, strength: f32 },
    Away { point: Vec2, strength: f32 },
    None,
}

impl Gravity {
    #[inline(always)]
    fn acceleration(&self, pos: Vec2) -> Vec2 {
        fn towards(point: Vec2, strength: f32, pos: Vec2) -> Vec2 {
            let axis = point - pos;
            let sqr_len = axis.length_squared();
            let len = sqr_len.sqrt();
            axis * (strength / (sqr_len * len))
        }
        match self {
            Gravity::Dir(dir) => *dir,
            Gravity::Towards { point, strength } => towards(*point, *strength, pos),
            Gravity::Away { point, strength } => -towards(*point, *strength, pos),
            Gravity::None => Vec2::ZERO,
        }
    }
}

#[derive(Clone)]
enum Bounds {
    Circle(f32),
    Rect(Vec2, Vec2),
    None,
}

impl Bounds {
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
    fn update_position(&self, obj: &mut Object) {
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
}

struct PhysWorld {
    objects: Vec<Object>,
    gravity: Gravity,
    bounds: Bounds,
}

impl PhysWorld {
    fn new(gravity: Gravity, bounds: Bounds) -> Self {
        Self {
            objects: Vec::new(),
            gravity,
            bounds,
        }
    }

    fn spawn_random(&mut self, rng: &mut impl Rng) {
        let material = match rng.gen_range(0..40) {
            0 => Material::Metal,
            1 => Material::Wood,
            _ => Material::Water,
        };
        let radius = match material {
            Material::Metal | Material::Wood => rng.gen_range(4.0..8.0),
            Material::Water => 1.5,
        };
        let pos = self.bounds.random_point(rng, radius);
        self.objects.push(Object {
            pos,
            pos_old: pos,
            acceleration: Vec2::ZERO,
            radius,
            material,
            binding: None,
        });
    }

    fn tick(&mut self, dt: f32) {
        // Handle gravity
        {
            #[cfg(feature = "tracy")]
            profiling::scope!("gravity");
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
            for i in 0..self.objects.len() {
                for j in 0..self.objects.len() {
                    let o0 = &self.objects[i];
                    let o1 = &self.objects[j];

                    let collision_axis = o0.pos - o1.pos;
                    let combined = o0.radius + o1.radius;
                    let dist_sqr = collision_axis.length_squared();
                    if i != j && dist_sqr < combined * combined {
                        let dist = dist_sqr.sqrt();
                        let n = collision_axis / dist;
                        let delta = combined - dist;
                        let w0 = o0.mass();
                        let w1 = o1.mass();
                        self.objects[i].pos += (w1 / (w0 + w1)) * delta * n;
                        self.objects[j].pos -= (w0 / (w0 + w1)) * delta * n;
                    }
                }
            }
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

struct Circle(Handle<Image>);

fn load_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(Circle(asset_server.load("circle.png")));
    const COUNT: usize = 2000;
    commands.insert_resource({
        let mut world = PhysWorld::new(Gravity::Dir(Vec2::new(0.0, -400.0)), Bounds::Circle(200.0));
        let mut rng = rand::thread_rng();
        for _ in 0..COUNT {
            world.spawn_random(&mut rng);
        }
        world
    });

    commands
        .spawn_bundle(OrthographicCameraBundle::new_2d())
        .insert(PanCam::default());
}

fn physics_system(mut phys_world: ResMut<PhysWorld>, time: Res<Time>) {
    #[cfg(feature = "tracy")]
    profiling::scope!("physics system");
    let sub_steps = 8;
    let dt = time.delta_seconds() / sub_steps as f32;
    for _ in 0..sub_steps {
        phys_world.tick(dt);
    }
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
            transform.translation = Vec3::new(o.pos.x, o.pos.y, transform.translation.z);
            transform.scale = Vec3::splat(o.radius * 2.0);
        } else {
            let e = commands
                .spawn_bundle(SpriteBundle {
                    sprite: Sprite {
                        color: o.material.color(),
                        custom_size: Some(Vec2::ONE),
                        ..default()
                    },
                    transform: Transform {
                        translation: Vec3::new(o.pos.x, o.pos.y, 0.0),
                        scale: Vec3::splat(o.radius * 2.0),
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

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(PanCamPlugin)
        .insert_resource(ClearColor(Color::BLACK))
        .add_startup_system(load_system)
        .add_system(physics_system)
        .add_system(extract_system)
        .run();
}
