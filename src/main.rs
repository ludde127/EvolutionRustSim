#![deny(clippy::all)]
#![forbid(unsafe_code)]

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

struct Pos {
    x: f32,
    y: f32,
}

struct Velocity {
    x: f32,
    y: f32,
}

struct Acceleration {
    x: f32,
    y: f32,
}

struct EvoObject {
    position: Pos,
    velocity: Velocity,
    acceleration: Acceleration,
}

impl EvoObject {
    fn new() -> Self {
        Self {
            position: Pos { x: 15.0, y: 20.0 },
            velocity: Velocity { x: 3.0, y: 2.0 },
            acceleration: Acceleration { x: 2.0, y: 3.0 }
        }
    }

    /// Update the `World` internal state; bounce the box around the screen.
    fn update(&mut self, dt: f32) {
        self.position.x = (self.position.x + self.velocity.x * dt) % WIDTH_AS_F32;
        self.position.y = (self.position.y + self.velocity.y * dt) % HEIGHT_AS_F32;
        self.velocity.x = self.velocity.x + self.acceleration.x * dt;
        self.velocity.y = self.velocity.y + self.acceleration.y * dt;
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
    let mut object = EvoObject::new();

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
