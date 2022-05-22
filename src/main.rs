use bevy::prelude::*;
use bevy_pancam::{PanCam, PanCamPlugin};
use itertools::Itertools;
use rand::Rng;

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

type Grid = Vec<Vec<Vec<Object>>>;

struct PhysWorld {
    grid_a: Grid,
    grid_b: Grid,
    current_grid: bool,
    tile_size: f32,
    width: usize,
    height: usize,
}

impl PhysWorld {
    fn new(width: usize, height: usize, tile_size: f32) -> Self {
        let grid = vec![vec![vec![]; width]; height];
        Self {
            grid_a: grid.clone(),
            grid_b: grid,
            current_grid: false,
            tile_size,
            width,
            height,
        }
    }
    fn grid(&self) -> &Grid {
        if self.current_grid {
            &self.grid_a
        } else {
            &self.grid_b
        }
    }
    fn grid_mut(&mut self) -> &mut Grid {
        if self.current_grid {
            &mut self.grid_a
        } else {
            &mut self.grid_b
        }
    }
    fn ngrid(&mut self) -> &Grid {
        if self.current_grid {
            &self.grid_b
        } else {
            &self.grid_a
        }
    }
    fn ngrid_mut(&mut self) -> &mut Grid {
        if self.current_grid {
            &mut self.grid_b
        } else {
            &mut self.grid_a
        }
    }

    fn grids_mut(&mut self) -> (&mut Grid, &mut Grid) {
        if self.current_grid {
            (&mut self.grid_a, &mut self.grid_b)
        } else {
            (&mut self.grid_b, &mut self.grid_a)
        }
    }
    fn spawn_random(&mut self, rng: &mut impl Rng) {
        let x = rng.gen_range(0.0..self.width as f32);
        let y = rng.gen_range(0.0..self.height as f32);
        let pos = Vec2::new(x, y) * self.tile_size;
        let material = match rng.gen_range(0..1) {
            0 => Material::Metal,
            1 => Material::Wood,
            _ => Material::Water,
        };
        self.grid_mut()[y as usize][x as usize].push(Object {
            pos,
            pos_old: pos,
            acceleration: Vec2::ZERO,
            radius: match material {
                Material::Metal | Material::Wood => rng.gen_range(4.0..8.0),
                Material::Water => 0.5,
            },
            material,
            binding: None,
        });
    }

    fn all(&self) -> impl Iterator<Item = &Object> {
        self.grid()
            .iter()
            .flat_map(|r| r.iter())
            .flat_map(|c| c.iter())
    }

    fn all_mut(&mut self) -> impl Iterator<Item = &mut Object> {
        self.grid_mut()
            .iter_mut()
            .flat_map(|r| r.iter_mut())
            .flat_map(|c| c.iter_mut())
    }

    fn update_positions(&mut self, mut updater: impl FnMut((usize, usize), &mut Object)) {
        let tile_size = self.tile_size;
        let (width, height) = (self.width, self.height);
        let (curr, next) = self.grids_mut();
        for (y, x) in (0..height).flat_map(|y| (0..width).map(move |x| (x, y))) {
            for mut obj in curr[y][x].drain(0..) {
                updater((x, y), &mut obj);
                let n_x = ((obj.pos.x / tile_size).max(0.0) as usize).min(width - 1);
                let n_y = ((obj.pos.y / tile_size).max(0.0) as usize).min(height - 1);
                next[n_y][n_x].push(obj);
            }
        }
        self.current_grid = !self.current_grid;
    }

    fn tick(&mut self, dt: f32) {
        let gravity = Vec2::new(0.0, -500.0);

        // Handle gravity
        self.all_mut().for_each(|obj| obj.accelerate(gravity));

        // Handle bounds
        let size = Vec2::new(self.width as f32, self.height as f32) * self.tile_size;
        self.update_positions(|_, obj| {
            let diff = obj.pos - size / 2.0;
            let dist = diff.length();
            if dist > size.x / 2.0 - obj.radius {
                obj.pos = size / 2.0 + diff.normalize() * (size.x / 2.0 - obj.radius);
            }
            // let r = Vec2::splat(obj.radius);
            // obj.pos = obj.pos.clamp(r, size - r);
        });

        // Handle collisions
        let tile_size = self.tile_size;
        let (width, height) = (self.width, self.height);
        for (y0, x0) in (0..height).flat_map(|y| (0..width).map(move |x| (x, y))) {
            for i in 0..self.grid()[y0][x0].len() {
                let dc = (self.grid()[y0][x0][i].radius * 2.0 / tile_size).ceil() as usize;
                for (y1, x1) in (y0.saturating_sub(dc)..y0.saturating_add(dc).min(height))
                    .cartesian_product(x0.saturating_sub(dc)..x0.saturating_add(dc).min(width))
                {
                    for j in 0..self.grid()[y1][x1].len() {
                        if i == j && x0 == x1 && y0 == y1 {
                            continue;
                        }
                        let o0 = &self.grid()[y0][x0][i];
                        let o1 = &self.grid()[y1][x1][j];

                        let collision_axis = o0.pos - o1.pos;
                        let dist = collision_axis.length();
                        let combined = o0.radius + o1.radius;
                        if dist < combined {
                            let n = collision_axis / dist;
                            let delta = combined - dist;
                            let w0 = o0.mass();
                            let w1 = o1.mass();
                            self.grid_mut()[y0][x0][i].pos += (w1 / (w0 + w1)) * delta * n;
                            self.grid_mut()[y1][x1][j].pos -= (w0 / (w0 + w1)) * delta * n;
                        }
                    }
                }
            }
        }

        // Update positions
        self.update_positions(|_, o| {
            o.update_position(dt);
        });
    }
}

struct Circle(Handle<Image>);

fn load_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(Circle(asset_server.load("circle.png")));
    const COUNT: usize = 500;
    commands.insert_resource({
        let mut world = PhysWorld::new(100, 100, 4.0);
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
    world.all_mut().for_each(|o| {
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
