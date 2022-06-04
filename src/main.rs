#![deny(clippy::all)]
#![forbid(unsafe_code)]

use std::time::{Duration, Instant};
use std::thread::sleep;
use std::ops::{Add, Div, Mul};
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

const WIDTH: u32 = 600;
const HEIGHT: u32 = 400;

const WIDTH_AS_F32: f32 = WIDTH as f32;
const HEIGHT_AS_F32: f32 = WIDTH as f32;
const PIXEL_SIZE: f32 = 16.0;

const DRAG_COEFFICIENT: f64 = 0.02;
const DENSITY: f64 = 997.0; // Kg/m3. waters density
const DRAG_X_DENSITY: f64 = drag_coefficient * density;

fn drag_force(velocity: f32, cross_sectional_area: f32) -> f32 {
    // TODO MAKE THIS MORE REALISTIC  https://en.wikipedia.org/wiki/Drag_(physics)
    let drag =
        0.5 * velocity * velocity * cross_sectional_area * (DRAG_X_DENSITY as f32);
    drag*drag/drag // Abs value
}

fn pair_drag_force(velocity: Pair, cross_sectional_area: Pair) -> Pair {
    // TODO MAKE THIS MORE REALISTIC  https://en.wikipedia.org/wiki/Drag_(physics)
    let drag =
        value(0.5) * velocity * velocity * cross_sectional_area * value((DRAG_X_DENSITY as f32));
    drag*drag/drag // Abs value
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

struct Material {
    density: f32,
}


#[derive(Debug, Copy, Clone, PartialEq)]
struct Pair {
    x: f32,
    y: f32,
}

fn value(val: f32) -> Pair  {Pair{x: val, y: val}} // Kinda dumb way to handle numbers but what ever.

impl Add for Pair {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: x + rhs.x,
            y: y + rhs.y,
        }
    }
}

impl Mul for Pair {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: x * rhs.x,
            y: y * rhs.y,
        }
    }
}

impl Div for Pair {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Divides self by the rhs Pair coordinate wise
        Self {
            x: x / rhs.x,
            y: y / rhs.y,
        }
    }
}

struct Particle {
    position: Pair,
    velocity: Pair,
    acceleration: Pair,
    material: Material,
    dt: Instant,
}

impl Particle {
    fn new(position: Pair, velocity: Pair, acceleration: Pair, material: Material) -> Self {
        Self {
            position,
            velocity,
            acceleration,
            material,
            dt: Instant::now(),
        }
    }

    fn update(&mut self) {

        let dt_pair = value(self.dt.elapsed().as_micros() as f32);

        self.acceleration = self.acceleration * value(self.material.density)
            - pair_drag_force(self.velocity, value(1.0));
        self.velocity = self.velocity + self.acceleration * dt_pair;


        self.dt = Instant::now(); // Reset the dt

    }
}

struct EvoObject {
    particle: Particle,
}

impl EvoObject {
    fn new(particle: Particle) -> Self {
        Self {
            particle
        }
    }

    fn default() -> Self {
        Self {
            particle: Particle {
                position: Pair { x: 35.0, y: 35.0},
                velocity: Pair { x: 10.0,  y: 10.0 },
                acceleration: Pair { x: 3.0,  y: 3.0},
                material: Material { density: 15.0},
                dt: Instant::now()
            }
        }
    }

    /// Update the `World` internal state; bounce the box around the screen.
    fn update(&mut self, dt: f32) {
        self.position.x = (self.position.x + self.velocity.x * dt) % WIDTH_AS_F32;
        self.position.y = (self.position.y + self.velocity.y * dt) % HEIGHT_AS_F32;

        let drag_x = self.universe.drag_force(self.velocity.x, PIXEL_SIZE as f32);
        let drag_y = self.universe.drag_force(self.velocity.y, PIXEL_SIZE as f32);
        if drag_x > self.acceleration.x {
            self.acceleration.x = 0.0;
        } else {
            self.acceleration.x = self.acceleration.x - drag_x;
        }
        if drag_y > self.acceleration.y {
            self.acceleration.y = 0.0;
        } else {
            self.acceleration.y = self.acceleration.y - drag_y;
        }
        self.velocity.x = self.velocity.x + self.acceleration.x * dt;
        self.velocity.y = self.velocity.y + self.acceleration.x * dt;
    }
    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let i = i as f32;
            let x = (i % WIDTH_AS_F32) as f32;
            let y = (i / WIDTH_AS_F32) as f32;

            let inside_the_box = x >= self.position.x
                && x < self.position.x + PIXEL_SIZE
                && y >= self.position.y
                && y < self.position.y + PIXEL_SIZE;

            let rgba = if inside_the_box {
                [0x5e, 0x48, 0xe8, 0xff]
            } else {
                [0x48, 0xb2, 0xe8, 0xff]
            };

            pixel.copy_from_slice(&rgba);
        }
        println!("{}, {}", self.position.x, self.position.y)
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

    let universe = Universe::new();
    let mut object = EvoObject::new(universe);

    event_loop.run(move |event, _, control_flow| {
        // Draw the current frame
        if let Event::RedrawRequested(_) = event {
            object.draw(pixels.get_frame());
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
            object.update(0.1);
            window.request_redraw();
        }
    });
}
