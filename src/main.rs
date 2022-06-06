#![deny(clippy::all)]
#![forbid(unsafe_code)]

use std::time::{Duration, Instant};
use std::thread::sleep;
use std::ops::{Add, Div, Mul, Range, Sub};
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::{LogicalSize, Position};
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;
use std::collections::{HashMap, HashSet};
use std::fmt;


const WIDTH: u32 = 600;
const HEIGHT: u32 = 400;

const WIDTH_AS_F64: f64 = WIDTH as f64;
const HEIGHT_AS_F64: f64 = HEIGHT as f64;
const PIXEL_SIZE: f64 = 8.0;

const DENSITY: f64 = 997.0; // Kg/m3. waters density
const DRAG_COEFFICIENT: f64 = 2.0;

struct RGBA {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

struct Pixel {
    x: u16,
    y: u16,
    color: RGBA,
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Material {
    density: f64,
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Geometry {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

impl Geometry {
    fn new(x0: u32, x1: u32, y0: u32, y1: u32) -> Self {
        Self {
            x0,
            x1,
            y0,
            y1
        }
    }

    fn __x_in_geometry(&self, x: &u32) -> bool {

        (&self.x0 <= x && x <= &self.x1) || // STANDARD CASE
            ((self.x1 % WIDTH < self.x0) && // This is true if it enters from right side
                (&(self.x1 % WIDTH) >= x && x >= &0))

    }

    fn __y_in_geometry(&self, y: &u32) -> bool {
        (&self.y0 <= y && y <= &self.y1) || // STANDARD CASE
            ((self.y1 % HEIGHT < self.y0) && // This is true if it enters from right side
                (&(self.y1 % HEIGHT) >= y && y >= &0))
    }
    
    fn contains(&self, x: &u32, y: &u32) -> bool {
        self.__x_in_geometry(x) && self.__y_in_geometry(y)
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.x1-self.x0, self.y1-self.y0)
    }

    fn area(&self) -> u32 {(self.x1-self.x0)*(self.y1-self.y0)}

    fn corners(&self) -> Vec<(u32, u32)> {
        let left_upper = (self.x0%WIDTH, self.y0%HEIGHT);
        let right_lower = (self.x1%WIDTH, self.y1%HEIGHT);
        let left_lower = (self.x0%WIDTH, self.y1%HEIGHT);
        let right_upper = (self.x1%WIDTH, self.y0%HEIGHT);
        vec![left_upper, right_lower, left_lower, right_upper]
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Pair {
    x: f64,
    y: f64,
}

impl Pair {
    fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y
        }
    }

    fn to_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    fn zeros() -> Self {
        Pair::new(0.0, 0.0)
    }

    fn value(val: f64) -> Self {
        Pair::new(val, val)
    }
    
    fn set_x(&mut self, x: f64) {
        self.x = x;
    }

    fn set_y(&mut self, y: f64) {
        self.y = y;
    }

    fn abs(&mut self) -> Self {self.x = self.x.abs(); self.y=self.y.abs(); self.clone()}

    fn to_dir(&self) -> Self {
        let mut temp = Pair::zeros();
        if self.x > 0.0 {
            temp.set_x(1.0);
        } else if self.x < 0.0 {
            temp.set_x(-1.0);
        }

        if self.y > 0.0 {
            temp.set_y(1.0);
        } else if self.y < 0.0 {
            temp.set_y(-1.0);
        }
        
        temp
    }
}

impl fmt::Display for Pair {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}, y: {}", self.x, self.y)
    }
}

fn value(val: f64) -> Pair  {Pair::value(val)} // Kinda dumb way to handle numbers but what ever.

impl Add for Pair {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Mul for Pair {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

impl Div for Pair {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Divides self by the rhs Pair coordinate wise
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
        }
    }
}

impl Sub for Pair {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Particle {
    position: Pair,
    velocity: Pair,
    acceleration: Pair,
    material: Material,
    occupied_pixels: Geometry,
    size: u32,
    dt: Instant,
    paused: bool,
    was_paused: bool,
}

impl Particle {
    fn new(position: Pair, velocity: Pair, acceleration: Pair, material: Material, occupied_pixels: Geometry, size: u32) -> Self {
        Self {
            position,
            velocity,
            acceleration,
            material,
            occupied_pixels,
            size,
            dt: Instant::now(),
            paused: false,
            was_paused: false,
        }
    }

    fn new_square(position: Pair, velocity: Pair, acceleration: Pair, material: Material, size: u32) -> Self {
        Self {
            position,
            velocity,
            acceleration,
            material,
            occupied_pixels: Geometry::new(position.x as u32, (position.x as u32)+size,
                                           position.y as u32, (position.y as u32)+size),
            size,
            dt: Instant::now(),
            paused: false,
            was_paused: false,
        }
    }

    fn new_still_square(x: f64, y:f64, side: u32) -> Self {
        Particle::new_square(Pair::new(x, y), Pair::zeros(),
                             Pair::zeros(), Material{density:20.0}, side)
    }

    fn drag_force(&self) -> Pair {
        value(0.5) * self.velocity.to_dir() * value((-1.0)) *
            self.velocity * self.velocity * value((self.size as f64) * DRAG_COEFFICIENT)
    }

    fn update(&mut self) {
        assert!(!self.paused, "Tried to update while paused");
        if !self.was_paused {
            let dt_pair = value(self.dt.elapsed().as_secs_f64());

            self.apply_force(self.drag_force());
            //println!("{}", self.position);
            self.velocity = self.velocity + self.acceleration * dt_pair;

            // Modulo position to wrap around window.
            self.position = self.position + self.velocity * dt_pair;
            self.position.x = self.position.x % WIDTH_AS_F64;
            self.position.y = self.position.y % HEIGHT_AS_F64;

            if self.position.x <= 0.0 {
                self.position.x = WIDTH_AS_F64;
            }

            if self.position.y <= 0.0 {
                self.position.y = HEIGHT_AS_F64;
            }
            self.acceleration = Pair::zeros();
            assert!(self.position.x >= 0.0 && self.position.y >= 0.0, "Position should be positive!");
            self.__generate_occupied();
        }
        self.was_paused = false;
        self.dt = Instant::now(); // Reset the dt
    }

    fn occupies(&self, x: &u32, y: &u32) -> bool {
        self.occupied_pixels.contains(x, y)
    }

    fn __generate_occupied(&mut self) {
        self.occupied_pixels = Geometry::new(self.position.x as u32, (self.position.x as u32)+self.size,
                                             self.position.y as u32, (self.position.y as u32)+self.size)
    }

    fn approximate_position(&self) -> (u32, u32) {
        (self.position.x as u32, self.position.y as u32)
    }

    fn set_position_from_approximate(&mut self, approximate_pos: (u32, u32)){
        self.position = Pair::new(approximate_pos.0 as f64, approximate_pos.1 as f64);
    }

    fn set_position(&mut self, position: Pair) {
        self.position = position;

    }

    fn apply_force(&mut self, force: Pair){
        self.acceleration = self.acceleration + force/value(self.mass());
    }

    fn toggle_pause(&mut self){
        self.paused = !self.paused;
        self.was_paused = !self.paused;
    }

    fn mass(&self) -> f64 {self.material.density * (self.size as f64)}

    fn collides_with(&self, other: &Particle) -> bool {
        self.occupied_pixels.corners().iter().any(|corner| other.occupies(&corner.0, &corner.1))
    }
}

struct Particles {
    particles: Vec<Particle>,
    paused: bool,
}


impl Particles {
    // This implements the behaviour for a group of particles.

    fn new_square(particle: Particle) -> Self {
        let mut p = vec![];
        p.push(particle);

        Self {
            particles: p.clone(),
            paused: false,
        }
    }

    fn new_still_square(x:f64, y:f64, side:u32) -> Self {
        Particles::new_square(Particle::new_still_square(x, y, side))
    }

    fn default() -> Self {
        Particles::new_square(Particle::new_square(
            Pair { x: 35.0, y: 35.0 },
            Pair { x: 15.0, y: 15.0 },
            Pair { x: 0.0, y: 0.0 },
            Material { density: 15.0 },
            32
        ))
    }

    fn precise_occupies(&self, x: &u32, y:&u32) -> Option<&Particle> {
        let mut found = None;
        for particle in &self.particles {
            if particle.occupies(x, y) {found = Some(particle); break}
        }
        found
    }

    fn any_occupies(&self, x: &u32, y: &u32) -> bool {
        self.particles.iter().any(|p|
            p.occupied_pixels.contains(x, y))
    }

    /// Update the state of the group of particles.
    fn update(&mut self) {
        self.particles.iter_mut().for_each(|f| f.update());
    }

    fn apply_force(&mut self, force: Pair){
        self.particles.iter_mut().for_each(|p| p.apply_force(force));
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.particles.iter_mut().for_each(|p| p.toggle_pause());
    }

    fn collides_with(&self, other: &Particles) -> bool {
        self.particles.iter().any(|p| other.particles.iter().any(|o| p.collides_with(o)))
    }
    
    fn velocity(&self) -> Pair {self.particles.get(0).unwrap().velocity} // All should have same speeds.
}

struct ObjectCoordinator {
    objects: Vec<Particles>,
    player_particles: Option<usize>,
    paused: bool,
}

impl ObjectCoordinator {
    fn new() -> Self {
        Self {
            objects: vec![],
            player_particles: None,
            paused: false,
        }
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let i = i as u32;
            let x = (i % WIDTH) as u32;
            let y = (i / WIDTH) as u32;
            if self.any_occupies(&x, &y) {
                pixel.copy_from_slice(&[0x5e, 0x48, 0xe8, 0xff]);
            } else {
                pixel.copy_from_slice(&[0x48, 0xb2, 0xe8, 0xff]);
            }
        }
    }

    fn any_occupies(&self, x: &u32, y: &u32) -> bool {
        self.objects.iter().any(|ps| ps.any_occupies(x, y))
    }

    fn collision_handling(&mut self) {
        // TODO MAKE THIS MORE EFFICIENT
        if &self.objects.len() > &1 {
            let mut checked_pairs: HashSet<(usize, usize)> = HashSet::new();
            let mut collisions: HashSet<(usize, usize)> = HashSet::new();
            for (n, particles) in self.objects.iter().enumerate() {
                for (n2, other) in self.objects.iter().enumerate() {
                    if n != n2 && !checked_pairs.contains(&(n, n2)) {
                        if particles.collides_with(other) {
                            collisions.insert((n, n2));
                        }
                        checked_pairs.insert((n, n2));
                        checked_pairs.insert((n2, n));
                    }
                }
            }
            for (n, n2) in collisions {
                let forces = self.objects.get(n).unwrap().velocity().clone()
                    * value(-1.0) * self.objects.get(n2).unwrap().velocity().clone();

                self.objects.get_mut(n).unwrap().apply_force(forces*value(100.0) * value(-1.0));
                self.objects.get_mut(n2).unwrap().apply_force(forces*value(100.0));

                println!("COLLISION");
            }
        }
    }

    fn precise_occupies(&self, x: &u32, y: &u32) -> Vec<&Particle> {
        let mut occupying = vec![];
        for ps in &self.objects {
            let occ = ps.precise_occupies(x, y);
            if occ.is_some() {
                occupying.push(occ.unwrap());
            }
        }
        occupying
    }

    fn add(&mut self, particles: Particles) {
        self.objects.push(particles);
    }

    fn apply_force_to_player(&mut self, force: Pair) {
        self.objects.get_mut(self.player_particles.unwrap()).unwrap().apply_force(force);
    }

    fn add_player(&mut self, particles: Particles) {
        self.player_particles = Some(self.objects.len());
        self.objects.push(particles);
    }

    fn update(&mut self) {
        if !self.paused {
            self.objects.iter_mut().for_each(|p| p.update());
            self.collision_handling();
        }
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.objects.iter_mut().for_each(|p| p.toggle_pause());
    }
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Evolution")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };

    let mut state = ObjectCoordinator::new();
    state.add(Particles::default());
    state.add_player(Particles::new_square(
        Particle::new_square(Pair { x: 50.0, y: 50.0 },
                      Pair {x: 20.0, y: 26.0},
                      Pair {x: 0.0, y: 0.0},
                      Material {density: 20.0}, 55),
    ));
    //state.add(Particles::new_still_square(WIDTH_AS_F64-12.0, HEIGHT_AS_F64-12.0, 24));
    //state.add(Particles::new_still_square(WIDTH_AS_F64-12.0, 12.0, 24));
    event_loop.run(move |event, _, control_flow| {
        // Draw the current frame
        if let Event::RedrawRequested(_) = event {
            state.draw(pixels.get_frame());
            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            } else if input.key_pressed(VirtualKeyCode::Space) {
                state.toggle_pause();
            }

            if state.player_particles.is_none() || state.paused {

            } else if input.key_held(VirtualKeyCode::Up) {
                state.apply_force_to_player(Pair::new(0.0, -100000.0))
            } else if input.key_held(VirtualKeyCode::Down) {
                state.apply_force_to_player(Pair::new(0.0, 100000.0))}
            else if input.key_held(VirtualKeyCode::Left) {
                state.apply_force_to_player(Pair::new(-100000.0, 0.0))}
            else if input.key_held(VirtualKeyCode::Right) {
                state.apply_force_to_player(Pair::new(100000.0, 0.0))}

            // Resize the window
            if let Some(size) = input.window_resized() {
                pixels.resize_surface(size.width, size.height);
            }

            // Update internal state and request a redraw

            state.update();
            window.request_redraw();


        }
    });
}
