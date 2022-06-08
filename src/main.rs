#![deny(clippy::all)]
#![forbid(unsafe_code)]
use std::time::{Instant};
use std::ops::{Add, Div, Mul, Sub};
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::{LogicalSize};
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
const WIDTH_HEIGHT: (f64, f64) = (WIDTH_AS_F64, HEIGHT_AS_F64);

const DENSITY: f64 = 997.0; // Kg/m3. waters density
const DRAG_COEFFICIENT: f64 = 2.0;
const PI: f64 = std::f64::consts::PI;

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
    elasticity: f64,
}

impl Material {
    fn default() -> Self {Material{density: 20.0, elasticity: 0.8}}
}

trait Geometry: std::fmt::Debug {
    fn contains(&self, x: &f64, y: &f64) -> bool;
    fn area(&self) -> f64;
    fn center(&self) -> Pair;
    fn _move(&mut self, x: f64, y: f64);
    fn position(&self) -> Pair;
    fn collided_with(&self, other: Shape) -> bool;
    fn shape(&self) -> Shape;
    fn modulo(&mut self);
    fn drag_area(&self) -> f64;
    fn corners(&self) -> Vec<(f64, f64)>;
}

#[derive(Debug, Clone, PartialEq)]
enum Shape {
    Rectangle(Rectangle),
    Circle(Circle),
}

fn circle_rectangle_intersect(circle: Circle, rectangle: Rectangle) -> bool {
    let center_dist = (circle.center()-rectangle.center()).abs();

    if center_dist.x > (rectangle.length/2.0 + circle.diameter/2.0) ||
        (center_dist.y > (rectangle.height/2.0 + circle.diameter/2.0)) {
        false
    } else if center_dist.x <= rectangle.length/2.0 || center_dist.y <= rectangle.height/2.0 {
        true
    } else {
        (center_dist-Pair::new(rectangle.length/2.0, rectangle.height/2.0)).pythagoras()
            <= (circle.diameter/2.0).powf(2.0)
    }
}

fn collided(shape_one: Shape, shape_two: Shape) -> bool {
    match shape_one {
        Shape::Rectangle(rectangle1) => {
            match shape_two {
                Shape::Rectangle(rectangle2) => {
                    rectangle1.corners().iter().any(|c| rectangle2.contains(&c.0, &c.1))
                }
                Shape::Circle(circle2) => {
                    circle_rectangle_intersect(circle2, rectangle1)
                }
            }
        },
        Shape::Circle(circle1) => {
            match shape_two {
                Shape::Rectangle(rectangle2) => {
                    circle_rectangle_intersect(circle1, rectangle2)
                },
                Shape::Circle(circle2) => {
                    circle1.center().distance(circle2.center()) <= circle1.diameter/2.0+circle2.diameter/2.0
                },
            }
        }
    }

}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Rectangle {
    x0: f64,
    y0: f64,
    length: f64,
    height: f64,
}

impl Rectangle {
    fn new(x0: f64, y0: f64, length: f64, height: f64) -> Self {
        Self {
            x0,
            y0,
            length,
            height
        }
    }

    fn new_square(x: f64, y: f64, side: f64) -> Self {
        Self {
            x0: x,
            y0: y,
            length: side,
            height: side,
        }
    }
    
    
    fn __x_in_geometry(&self, x: &f64) -> bool {
        (&self.x0 <= x && x <= &(self.x0+self.length)) || // STANDARD CASE
            (((self.x0+self.length) % WIDTH_AS_F64 < self.x0) && // This is true if it enters from right side
                (&((self.x0+self.length) % WIDTH_AS_F64) >= x && x >= &0.0))

    }

    fn __y_in_geometry(&self, y: &f64) -> bool {
        (&self.y0 <= y && y <= &(self.y0+self.height)) || // STANDARD CASE
            (((self.y0+self.height) % HEIGHT_AS_F64 < self.y0) && // This is true if it enters from right side
                (&((self.y0+self.height) % HEIGHT_AS_F64) >= y && y >= &0.0))
    }
}

impl Geometry for Rectangle {
    fn contains(&self, x: &f64, y: &f64) -> bool {
        // First check should be faster than lower checks.
        self.__x_in_geometry(x) && self.__y_in_geometry(y)
    }
    
    fn area(&self) -> f64 {((self.x0+self.length)-self.x0)*((self.y0+self.height)-self.y0)}
    
    fn center(&self) -> Pair {
        Pair::new((((self.x0+self.length) - self.x0) as f64)/2.0, (((self.y0+self.height) - self.y0) as f64)/2.0)
    }

    fn _move(&mut self, x: f64, y: f64) {
        self.x0 = x;
        self.y0 = y;
    }

    fn position(&self) -> Pair {
        Pair::new(self.x0, self.y0)
    }

    fn collided_with(&self, other: Shape) -> bool {
        collided(self.shape(), other)
    }

    fn shape(&self) -> Shape {
        Shape::Rectangle(*self)
    }

    fn modulo(&mut self) {
        (self.x0, self.y0) = Pair::new(self.x0, self.y0).modulo_window().to_tuple();
    }

    fn drag_area(&self) -> f64 {
        (self.length+self.height)*0.5
    }

    fn corners(&self) -> Vec<(f64, f64)> {
        let left_upper = (self.x0, self.y0);
        let right_lower = ((self.x0+self.length)%WIDTH_AS_F64, (self.y0+self.height)%HEIGHT_AS_F64);
        let left_lower = (self.x0, (self.y0+self.height)%HEIGHT_AS_F64);
        let right_upper = ((self.x0+self.length)%WIDTH_AS_F64, self.y0);
        vec![left_upper, right_lower, left_lower, right_upper]
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Circle {
    x: f64, // Center
    y: f64, // Center
    diameter: f64,
}

impl Circle {
    fn new(x: f64, y: f64, diameter: f64) -> Self {
        Self { x, y, diameter}
    }
}

impl Geometry for Circle {
    fn contains(&self, x: &f64, y: &f64) -> bool {
        // TODO FIX THIS SO IT WRAPS
        let given = Pair::new(*x, *y);
        let diff = self.center() - given;
        let normal =  diff.pythagoras()<=self.diameter/2.0; // True if they lay in same modulo plane and intersect
        normal
    }

    fn area(&self) -> f64 {
        (self.diameter/2.0).powf(2.0)*PI
    }

    fn center(&self) -> Pair {
        Pair::new(self.x, self.y)
    }

    fn _move(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
    }

    fn position(&self) -> Pair {
        self.center()
    }

    fn collided_with(&self, other: Shape) -> bool {
        collided(self.shape(), other)
    }

    fn shape(&self) -> Shape {
        Shape::Circle(*self)
    }

    fn modulo(&mut self) {
        (self.x, self.y) = Pair::new(self.x, self.y).modulo_window().to_tuple();
    }

    fn drag_area(&self) -> f64 {
        self.diameter
    }

    fn corners(&self) -> Vec<(f64, f64)> {
        /// Cornors of the smallest square which can fit the circle.
        let x0 = self.x - self.diameter/2.0;
        let x1 = self.x + self.diameter/2.0;
        let y0 = self.y - self.diameter/2.0;
        let y1 = self.y + self.diameter/2.0;

        let left_upper = (x0, y0);
        let right_lower = (x1%WIDTH_AS_F64, y1%HEIGHT_AS_F64);
        let left_lower = (x0, y1%HEIGHT_AS_F64);
        let right_upper = (x1%WIDTH_AS_F64, y0);
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

    fn width_height() -> Self {
        Pair::new(WIDTH_AS_F64, HEIGHT_AS_F64)
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

    fn modulo(&self, b: Pair) -> Self {
        let mut x = self.x%b.x;
        let mut y = self.y%b.y;
        if x < 0.0 {x = x + b.x};
        if y < 0.0 {y = y + b.y};
        Pair::new(x, y) // This should have the behaviour of normal modulo operations in math.
    }

    fn modulo_window(&self) -> Self { // Shorthand for fitting to the window as this is very common.
        self.modulo(Pair::width_height())
    }

    fn squared(&self) -> Self {Pair::new(self.x*self.x, self.y*self.y)}

    fn distance(&self, other: Pair) -> f64 {
        ((self.x-other.x).powf(2.0)+(self.y-other.y).powf(2.0)).sqrt()
    }

    fn pythagoras(&self) -> f64 {
        (self.x.powf(2.0) + self.y.powf(2.0)).sqrt()
    }

    fn cross_prod(&self, other: Pair) -> f64 {
        self.y*other.x - self.x*other.y
    }

    fn dot_prod(&self, other: Pair) -> f64 {
        self.x*other.y + self.y*other.x
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

impl Mul<f64> for Pair {
    type Output = Pair;

    fn mul(self, rhs: f64) -> Pair {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Pair> for f64 {
    type Output = Pair;

    fn mul(self, rhs: Pair) -> Pair {
        rhs * self
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

#[derive(Debug)]
struct Particle {
    geometry: Box<dyn Geometry>,
    velocity: Pair,
    acceleration: Pair,
    material: Material,
    dt: Instant,
    paused: bool,
    was_paused: bool,
}
impl Particle {
    fn new(geometry: Box<dyn Geometry>, velocity: Pair, acceleration: Pair, material: Material) -> Self {
        Self {
            geometry,
            velocity,
            acceleration,
            material,
            dt: Instant::now(),
            paused: false,
            was_paused: false,
        }
    }

    fn new_square(position: Pair, velocity: Pair, acceleration: Pair, material: Material, height: f64) -> Self {
        Self {
            geometry: Box::new(Rectangle::new(position.x, position.x + height,
                                                      position.y, position.y + height)),
            velocity,
            acceleration,
            material,
            dt: Instant::now(),
            paused: false,
            was_paused: false,
        }
    }
    
    fn new_still_rectangle(x: f64, y: f64, length: f64, height: f64) -> Self {
        Self {
            geometry: Box::new(Rectangle::new(x, x + length,
                                                      y, y + height)),
            velocity: Pair::zeros(),
            acceleration: Pair::zeros(),
            material: Material::default(),
            dt: Instant::now(),
            paused: false,
            was_paused: false
        }
    }
    
    fn new_still_square(x: f64, y:f64, side: f64) -> Self {
        Particle::new_square(Pair::new(x, y), Pair::zeros(),
                             Pair::zeros(), Material::default(), side)
    }

    fn new_still_circle(x: f64, y: f64, diameter: f64) -> Self {
        Self {
            geometry: Box::new(Circle::new(x, y, diameter)),
            velocity: Pair::zeros(),
            acceleration: Pair::zeros(),
            material: Material::default(),
            dt: Instant::now(),
            paused: false,
            was_paused: false,
        }
    }

    fn drag_force(&self) -> Pair {
        value(0.5) * self.velocity.to_dir() * value((-1.0)) *
            self.velocity * self.velocity * value(self.geometry.drag_area() * DRAG_COEFFICIENT)
    }

    fn update(&mut self) {
        assert!(!self.paused, "Tried to update while paused");
        if !self.was_paused {
            let dt_pair = value(self.dt.elapsed().as_secs_f64());

            self.apply_force(self.drag_force());
            //println!("{}", self.position);
            self.velocity = self.velocity + self.acceleration * dt_pair;

            // Modulo position to wrap around window.
            let pos = self.geometry.position() + self.velocity * dt_pair;
            self.geometry._move(pos.x%WIDTH_AS_F64, pos.y%HEIGHT_AS_F64);
            self.acceleration = Pair::zeros();
        }
        self.was_paused = false;
        self.dt = Instant::now(); // Reset the dt
    }

    fn occupies(&self, x: &f64, y: &f64) -> bool {
        self.geometry.contains(x, y)
    }

    fn apply_force(&mut self, force: Pair){
        self.acceleration = self.acceleration + force/value(self.mass());
    }

    fn set_velocity(&mut self, velocity: Pair) {
        self.velocity = velocity;
        self.acceleration = Pair::zeros(); // This is needed as the velocity was forcefully changed
    }

    fn toggle_pause(&mut self){
        self.paused = !self.paused;
        self.was_paused = !self.paused;
    }

    fn mass(&self) -> f64 {self.material.density * self.geometry.area()}

    fn collides_with(&self, other: &Particle) -> bool {
        self.geometry.collided_with(other.geometry.shape())
    }
}

fn assign_to_quadrants(shape: &Box<dyn Geometry>) -> HashSet<(u8, u8)> {
    let hs: HashSet<(u8, u8)> = HashSet::from_iter(shape.corners().iter()
        .map(|cor| (((8.0*(cor.0)/WIDTH_AS_F64) as u8),
                    ((6.0*(cor.1)/HEIGHT_AS_F64) as u8)))
        );
    hs
}

struct ObjectCoordinator {
    objects: Vec<Particle>,
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
            let x = (i % WIDTH) as f64;
            let y = (i / WIDTH) as f64;
            if self.any_occupies(&x, &y) {
                pixel.copy_from_slice(&[0x5e, 0x48, 0xe8, 0xff]);
            } else {
                pixel.copy_from_slice(&[0x48, 0xb2, 0xe8, 0xff]);
            }
        }
    }

    fn any_occupies(&self, x: &f64, y: &f64) -> bool {
        self.objects.iter().any(|ps| ps.occupies(x, y))
    }

    fn collision_handling(&mut self) {
        // TODO MAKE THIS MORE EFFICIENT
        if &self.objects.len() > &1 {
            let mut checked_pairs: HashSet<(usize, usize)> = HashSet::new();
            let mut collisions: HashSet<(usize, usize)> = HashSet::new();

            // Create the quadrants with the index of the particles in the storage vector instead.
            let quadrant_assignment: HashMap<Pair, HashSet<(u8, u8)>> = HashMap::from_iter(self.objects.iter().map(|p| (p.geometry.position(), assign_to_quadrants(&p.geometry))));

            for (n, particle) in self.objects.iter().enumerate() {
                for (n2, other) in self.objects.iter().enumerate() {
                    if n != n2 && !checked_pairs.contains(&(n, n2))
                        && !quadrant_assignment[&particle.geometry.position()].
                            is_disjoint(&quadrant_assignment[&other.geometry.position()]){
                        if particle.collides_with(other) {
                            collisions.insert((n, n2));
                        }
                        checked_pairs.insert((n, n2));
                        checked_pairs.insert((n2, n));
                    }
                }
            }
            for (n, n2) in collisions {
                let obj1 = self.objects.get(n).unwrap();
                let obj2 = self.objects.get(n2).unwrap();
                let dist = obj1.geometry.center().distance(obj2.geometry.center());
                let v1 = obj1.velocity;
                let v2 = obj2.velocity;

                let m1 = obj1.mass();
                let m2 = obj2.mass();

                let sys_mass = m1 + m2;
                if dist < 15.0 {
                    // Merge them!
                    let v_result = (m1*v1 + m2*v2)*(1.0/sys_mass);


                    let was_player = self.player_particles.is_some() &&
                        (self.player_particles.unwrap() == n ||
                            self.player_particles.unwrap() == n2);

                    let mean_pos = (obj1.geometry.position() + obj2.geometry.position()) * 0.5;

                    let d3 = 2.0*((obj1.geometry.area() + obj2.geometry.area())/PI).sqrt(); // r**2*pi = A r**2=A/pi r = sqrt(A/pi)

                    let new = Particle::new(
                        Box::new(Circle::new(mean_pos.x, mean_pos.y, d3)),
                        v_result, Pair::zeros(), obj1.material);
                    self.objects.remove(n);
                    self.objects.remove(n2-1);

                    if was_player {
                        self.add_player(new);
                    } else {
                        self.add(new);
                    }

                } else {
                    let cr1 = 0.95*obj1.material.elasticity/
                        (obj1.material.elasticity+obj2.material.elasticity);
                    let cr2 = 0.95*obj2.material.elasticity/
                        (obj1.material.elasticity+obj2.material.elasticity);
                    let common_part = m1*v1 + m2*v2;
                    let res1 = (common_part + m2*cr1 * (v2-v1))*(1.0/sys_mass);
                    let res2 = (common_part + m1*cr2 * (v1-v2))*(1.0/sys_mass);
                    self.objects.get_mut(n).unwrap().set_velocity(res1);
                    self.objects.get_mut(n2).unwrap().set_velocity(res2);
                }
            }
        }
    }

    fn precise_occupies(&self, x: &f64, y: &f64) -> Vec<&Particle> {
        let mut occupying = vec![];
        for p in &self.objects {
            let occ = p.occupies(x, y);
            if occ {
                occupying.push(p);
            }
        }
        occupying
    }

    fn add(&mut self, particle: Particle) {
        self.objects.push(particle);
    }

    fn apply_force_to_player(&mut self, force: Pair) {
        self.objects.get_mut(self.player_particles.unwrap()).unwrap().apply_force(force);
    }

    fn add_player(&mut self, particle: Particle) {
        self.player_particles = Some(self.objects.len());
        self.objects.push(particle);
    }

    fn update(&mut self) {
        if !self.paused {
            self.objects.iter_mut().for_each(|p| {p.update(); p.geometry.modulo();});
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
        let window_width = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_width.width, window_width.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };

    let mut state = ObjectCoordinator::new();

    state.add_player(Particle::new_still_circle(100.0, 200.0, 50.0));
    state.add(Particle::new_still_circle(200.0, 300.0, 50.0));
    state.add(Particle::new_still_circle(100.0, 100.0, 50.0));
    state.add(Particle::new_still_circle(50.0, 400.0, 50.0));
    state.add(Particle::new_still_circle(300.0, 124.0, 50.0));


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
                state.apply_force_to_player(Pair::new(0.0, -10000000.0))
            } else if input.key_held(VirtualKeyCode::Down) {
                state.apply_force_to_player(Pair::new(0.0, 10000000.0))}
            else if input.key_held(VirtualKeyCode::Left) {
                state.apply_force_to_player(Pair::new(-10000000.0, 0.0))}
            else if input.key_held(VirtualKeyCode::Right) {
                state.apply_force_to_player(Pair::new(10000000.0, 0.0))}

            // Rewidth the window
            if let Some(width) = input.window_resized() {
                pixels.resize_surface(width.width, width.height);
            }

            // Update internal state and request a redraw

            state.update();
            window.request_redraw();
        }
    });
}
