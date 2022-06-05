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
use std::collections::HashMap;
use std::fmt;


const WIDTH: u32 = 600;
const HEIGHT: u32 = 400;

const WIDTH_AS_F64: f64 = WIDTH as f64;
const HEIGHT_AS_F64: f64 = WIDTH as f64;
const PIXEL_SIZE: f64 = 8.0;

const DRAG_COEFFICIENT: f64 = 0.0002;
const DENSITY: f64 = 997.0; // Kg/m3. waters density
const DRAG_X_DENSITY: f64 = DRAG_COEFFICIENT * DENSITY;


fn drag_force(velocity: f64, cross_sectional_area: f64) -> f64 {
    // TODO MAKE THIS MORE REALISTIC  https://en.wikipedia.org/wiki/Drag_(physics)
    0.5 * velocity * velocity * cross_sectional_area * (DRAG_X_DENSITY as f64)

}

fn pair_drag_force(velocity: Pair, cross_sectional_area: Pair) -> Pair {
    // TODO MAKE THIS MORE REALISTIC  https://en.wikipedia.org/wiki/Drag_(physics)
    value(0.5) * velocity * velocity * cross_sectional_area * value((DRAG_X_DENSITY as f64))
}

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
    occupied_pixels: (Range<u32>, Range<u32>),
    size: u32,
    dt: Instant,
}

fn upscale(x: u32, y:u32, scale: u32) -> (Range<u32>, Range<u32>) {
    ((x..(x+scale)), (y..(y+scale))) // TODO: FIX ALL THE MODULU SHIT WHEN IT GOES OVER AN EDGE.
}

impl Particle {
    fn new(position: Pair, velocity: Pair, acceleration: Pair, material: Material, size: u32) -> Self {
        Self {
            position,
            velocity,
            acceleration,
            material,
            occupied_pixels: upscale(position.x as u32, position.y as u32, size),
            size,
            dt: Instant::now(),
        }
    }

    fn __drag_dir(&self) -> Pair {
        let mut d_dir = Pair::zeros();
        if self.velocity.x < 0.0 {
            d_dir.set_x(1.0);
        } else {d_dir.set_x(-1.0);}
        if self.velocity.y < 0.0 {
            d_dir.set_y(1.0);
        } else {d_dir.set_y(-1.0);}
        d_dir
    } 
    
    fn update(&mut self) {

        let dt_pair = value(self.dt.elapsed().as_secs_f64());
        let mass = value(self.material.density) * value(self.size as f64);
        
        let force = self.acceleration * mass + self.__drag_dir() * pair_drag_force(self.velocity.abs(), value(1.0));
        self.acceleration = force / mass; // F=ma => a = F/m
        //println!("{}", self.position);
        self.velocity = self.velocity + self.acceleration * dt_pair;

        // Modulo position to wrap around window.
        self.position = self.position + self.velocity * dt_pair;
        self.position.x = self.position.x % WIDTH_AS_F64;
        self.position.y = self.position.y % HEIGHT_AS_F64;

        assert!(self.position.x > 0.0 && self.position.y > 0.0, "Position should be positive!");
        self.__generate_occupied();
        self.dt = Instant::now(); // Reset the dt

    }

    fn occupies(&self, x: &u32, y: &u32) -> bool {
        self.occupied_pixels.0.contains(x) && self.occupied_pixels.1.contains(y)
    }

    fn __generate_occupied(&mut self) {
        self.occupied_pixels =
            upscale(self.position.x as u32, self.position.y as u32, self.size)
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
}

struct Particles {
    particles: Vec<Particle>,
    all_occupied_pixels: Vec<(Range<u32>, Range<u32>)>,
}


impl Particles {
    // This implements the behaviour for a group of particles.

    fn new_square(particle: Particle) -> Self {
        let mut p = vec![];
        p.push(particle);

        Self {
            particles: p.clone(),
            all_occupied_pixels: p.clone().iter().map(|p| p.occupied_pixels.clone()).collect(),
        }
    }

    fn default() -> Self {
        Particles::new_square(Particle::new(
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
        self.all_occupied_pixels.iter().any(|(r_x, r_y)|
            r_x.contains(x) && r_y.contains(y))
    }

    fn __generate_occupied(&mut self) {
        self.all_occupied_pixels = self.particles.iter().map(|p| p.occupied_pixels.clone()).collect();
    }

    /// Update the state of the group of particles.
    fn update(&mut self) {
        self.particles.iter_mut().for_each(|f| f.update());
        self.__generate_occupied();

    }

}

struct ObjectCoordinator {
    objects: Vec<Particles>,
}

impl ObjectCoordinator {
    fn new() -> Self {
        Self {
            objects: vec![],
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

    fn update(&mut self) {
        self.objects.iter_mut().for_each(|p| p.update());
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
            }

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
